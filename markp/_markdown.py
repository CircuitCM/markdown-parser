import re
from collections import OrderedDict
from typing import Iterable, Collection
from functools import cached_property
from itertools import islice
#import tiktoken

GPTTOKEN=None # tiktoken.get_encoding("cl100k_base")


_FIND_HDR=re.compile(r'^(?:#{1,6} .+)$',re.MULTILINE)
_VALUE_SUB=re.compile(r'#{1,6}\s*[\d.]*\s*')
_lol=re.compile('\n')
_LINE_IDX = re.compile(r'^[lL](\d+)$')

_rm_all_links = re.compile(r'(?<=\]\()(.*?:.*?)(?=\))|(?<=:\s)(.*?:.*?)(?=\s|$)|(?<=<)(.*?:.*?)(?=>)')
_rm_top_bottom_whtspc = re.compile(r'^\s+|\s+$')

class MDProcessors:
    redact_links = lambda md, txt='link': _rm_all_links.sub(txt, md)
    clean_whitespace = lambda md:_rm_top_bottom_whtspc.sub('',md)


class MarkdownIndexer:

    __slots__ = ('md','idx_max','idx_min', 'index','lindex')

    def __init__(self, input_data):
        """
        Build a markdown-aware index over headings and lines.
        
        This constructor accepts either a markdown string, a filesystem path to a
        markdown file, or a file-like object with ``.read()``. It parses all ATX
        headings (``#``–``######``) to produce a stable, numeric section index and a
        character-accurate line index, which power slicing, section selection, and
        post-processing.
        
        **Heading indexing**
        
        - Each heading’s level is inferred from the number of leading ``#`` characters.
        - A running counter is maintained at every level; deeper counters reset when a
          shallower counter increments (e.g., ``2.3`` → next sibling becomes ``2.4``,
          first child of ``2.4`` becomes ``2.4.1``).
        - The public ``index`` maps a dotted key (e.g., ``"2.4.1"``) to a tuple
          ``(title_text, start_char, end_char)`` where:
          - ``title_text`` is the heading line with the ``#...`` prefix and any
            leading numbering like ``"2.4.1"`` removed.
          - ``start_char`` is the character offset (0-based) where the section begins
            (at the heading’s ``#``).
          - ``end_char`` is the character offset where the section ends (start of the
            next section or the end of the document).
        
        - If the document does not begin with a heading (or the first heading is
          unusually deep relative to the minimum level found), a synthetic
          ``"0"`` section named ``"Start Placeholder"`` is inserted from character
          ``0`` up to the first real heading. This preserves preamble text as an
          addressable section.
        
        **Line indexing**
        
        - ``lindex`` is a list of ``(line_start, line_end)`` half-open spans per line.
          ``line_end`` points just past the newline so that ``md[line_start:line_end]``
          reconstructs the printed line reliably.
        
        The following attributes are set:
        
        - ``md``: the raw markdown text.
        - ``index``: ordered mapping of dotted keys to ``(title, start, end)``.
        - ``idx_min`` / ``idx_max``: the minimum heading level present and the count of
          distinct levels represented (useful for re-leveling later).
        - ``lindex``: per-line character spans as described above.
        
        :param input_data: A markdown string, a path to a markdown file, or a file-like
                           object with a ``read()`` method.
        :returns: A fully initialized indexer instance.
        :raises RuntimeError: If ``input_data`` is neither a string nor a readable
                              stream.
        :raises FileNotFoundError: If a string path is given and cannot be opened.
        
        **Examples**
        
         .. code-block:: python
        
            # From a path
            mdi = MarkdownIndexer("notes.md")
        
            # From a markdown string
            mdi = MarkdownIndexer("# Title\n\nContent...")
        
            # From a file-like object
            with open("notes.md", "r", encoding="utf8") as fh:
                mdi = MarkdownIndexer(fh)
        """
        if isinstance(input_data, str):
            if input_data.rfind('\n')==-1:
                with open(input_data, 'r',encoding="utf8") as f:
                    self.md = f.read()
            else:
                self.md = input_data
        elif hasattr(input_data, 'read'):
            self.md = input_data.read()
        else:
            raise
        #header indexing
        self.index = OrderedDict()
        lst1 = [(m.group(), m.start()) for m in _FIND_HDR.finditer(self.md)]
        lst2 = [x.count('#', 0, 6) for x, _ in lst1]
        mn, mx = min(lst2), max(lst2)
        self.idx_max=mx-mn+1
        self.idx_min=mn
        sec_ct = [0] * (mx - mn + 1)
        lst2 = [x - mn for x in lst2]
        if lst1[0][1] > 0 or lst2[0] > 10:
            sec_ct[0] = -1
            lst1.insert(0, ('Start Placeholder', 0))
            lst2.insert(0, 0)
        lld=len(lst2)-1
        for id, lvl in enumerate(lst2):
            sec_ct[lvl] += 1
            sec_ct[lvl + 1:] = [0] * (len(sec_ct) - lvl - 1)
            key = '.'.join(map(str, sec_ct[:lvl + 1]))
            self.index[key] = (_VALUE_SUB.sub( '', lst1[id][0]).strip(), lst1[id][1],lst1[id+1][1] if id<lld else len(self.md))

        #line indexing
        idxs = [m.start() for m in _lol.finditer(self.md)]
        self.lindex=[(idxs[i - 1]+1 if i else 0, idx+1) for i, idx in enumerate(idxs)] + ([(idxs[-1], len(self.md))] if idxs and idxs[-1]+1<len(self.md) else [])

    def fix_write(self,path:str=None,order_sections:bool=True,shift_headings:bool=False,mdprocessors=(MDProcessors.clean_whitespace,)):
        """
        Rewrite headers (and optionally normalize levels) using the computed index,
        optionally run post-processors, and return a new ``MarkdownIndexer`` (and/or
        write to disk).
        
        There are two independent operations:
        
        1) **Section ordering/numbering (``order_sections=True``)**  
           Replaces the first heading token in each indexed section with a canonical
           header that encodes both the level and the dotted section key produced by
           this index. For example, a second-level section with key ``"3.2"`` becomes
           ``"## 3.2. "`` regardless of the original spelling in the file. The body of
           the section is preserved byte-for-byte after that first heading line.
        
        2) **Level normalization (``shift_headings``)**  
           Computes a level shift so the shallowest heading either:
           - **is preserved** (default): when ``shift_headings=False``, the top-level
             headings in the output keep their original minimum level (e.g., if the
             shallowest heading was ``###``, top-level remains ``###``).
           - **becomes H1**: when ``shift_headings=True``, the shallowest headings are
             shifted to ``#`` and deeper levels are shifted accordingly.
        
        If ``order_sections=False`` but ``shift_headings`` is enabled, only the number
        of ``#`` characters in the very first heading line of each section is adjusted
        (one replacement per section), and **no** dotted numbering is injected.
        
        After rewriting, ``mdprocessors`` (if provided) are applied to the entire
        document in order. By default, leading/trailing file whitespace is trimmed.
        
        A new ``MarkdownIndexer`` over the rewritten markdown is returned (the original
        instance is left unchanged). If ``path`` is given, the rewritten text is also
        written there.
        
        :param path: Optional output path. When provided, the rewritten markdown is
                     saved to this location.
        :param order_sections: When ``True``, replace the first heading of each section
                               with a canonical one based on the dotted key (e.g.,
                               ``## 1.2. ``). When ``False``, headings are not
                               renumbered, only level-shifted if requested.
        :param shift_headings: When ``True``, normalize the shallowest heading level to
                               H1 and shift others accordingly. When ``False``, retain
                               the document’s original minimum level.
        :param mdprocessors: A callable or iterable of callables ``fn(str)->str`` that
                             are applied to the rewritten markdown in sequence. Use
                             ``None`` to disable post-processing.
        :returns: A ``MarkdownIndexer`` over the rewritten markdown (possibly the same
                  object if no changes were requested), and writes to ``path`` if set.
        
        **Examples**
        
        .. code-block:: python
        
            # Normalize so the shallowest heading becomes H1 and inject dotted numbers
            new = mdi.fix_write(order_sections=True, shift_headings=True)
        
            # Only shift levels (keep original titles and don’t inject "1.2.")
            new = mdi.fix_write(order_sections=False, shift_headings=True)
        
            # Write to disk and redact links via a custom processor
            new = mdi.fix_write(
                path="clean.md",
                order_sections=True,
                mdprocessors=(MDProcessors.redact_links, MDProcessors.clean_whitespace),
            )
        """
        # Initialize an empty list to store the updated markdown sections
        new_md=self.md
        if order_sections or shift_headings:
            updated_sections = []
            level_shift = 0 if shift_headings else self.idx_min - 1
            for key, (_, start_idx, end_idx) in self.index.items():
                section_md = self.md[start_idx:end_idx]
                k = len(key.split('.'))
                new_header_level = k + level_shift  # Calculate the new header level
                if order_sections:
                    new_header = f"{'#' * new_header_level} {key}. "  # Create the new header
                    section_md = _VALUE_SUB.sub(new_header, section_md, count=1)  # Replace only the first occurrence
                else:
                    #new_header = f"{'#' * new_header_level}"
                    section_md= section_md.replace('#'*(self.idx_min+k-1),'#'*new_header_level,1)
                updated_sections.append(section_md)
            new_md = ''.join(updated_sections)
        if mdprocessors is not None:
            if not isinstance(mdprocessors,Iterable):
                new_md=mdprocessors(new_md)
            else:
                new_md=apply_filters(new_md,*mdprocessors)

        if order_sections or shift_headings or mdprocessors is not None:
            nmd = MarkdownIndexer(new_md)
        else:
            nmd=self
        if path:
            with open(path, 'w') as f:
                f.write(new_md)

        return nmd


    def __str__(self):
        #will never call this much more than for user visualization
        pls = [f'{"   "*k.count(".")}{k} : {v[0]}' for k,v in self.index.items()]
        pls.append(f'Markdown is {len(self.lindex)} lines long, with total character length {len(self.md)}.')
        return '\n'.join(pls)

    def __repr__(self):
        return self.__str__()
    
    @property
    def info(self):
        """
        Summarize all indexed sections with both line and character extents.

        Each line of the returned string has the form::

            <key> - [L<start_line>:L<end_line>] [<start_char>:<end_char>] - <title>

        where:

        - ``<key>`` is the dotted section key from ``self.index`` (e.g. ``"1"``).
        - ``[Lstart:Lend]`` is a half-open range of **zero-based** line indices:
          - ``Lstart`` is the line where the section's heading begins.
          - ``Lend`` is the line where the section's end boundary falls
            (usually the next section's heading line, or ``L<N>`` for the end
            of the document).
        - ``[start:end]`` is the corresponding half-open character span from
          the index (``start`` inclusive, ``end`` exclusive).
        - ``<title>`` is the cleaned heading text for the section.

        This uses the precomputed ``index`` and ``lindex`` along with
        ``_find_line_idx``; it does not scan the markdown content.

        :returns: A newline-joined string describing all sections in document
                  order.
        """
        def _char_to_line(idx: int) -> int:
            # Map a character offset to a zero-based line index in lindex,
            # using len(lindex) as the open-ended sentinel for EOF.
            if idx >= len(self.md):
                return len(self.lindex)
            return self._find_line_idx(idx)

        #it might be a bit quicker to just build KV's for line to char to section. However 
        #_find_line_idx honestly shouldn't be that much slower.
        lines = []
        for key, (title, start_idx, end_idx) in self.index.items():
            start_line = _char_to_line(start_idx)
            end_line = _char_to_line(end_idx)
            lines.append(f"{'   '*key.count('.')}{key} - [L{start_line}:L{end_line}] [{start_idx}:{end_idx}] - {title}")
        return "\n".join(lines)
        

    def __getitem__(self, keys):
        """
        Return markdown substrings or annotated line views using flexible, pythonic
        indexing.
        
        This accessor accepts:
        
        - **Dotted keys** (e.g., ``"2.4.1"``): select that entire section.
        - **Integer character offsets**: select raw character spans.
        - **Line keys** of the form ``"L<n>"`` / ``"l<n>"``: map to the start/end
          character offsets of line ``n`` using the precomputed ``lindex``.
        - **Slices**: ``start:stop:step`` where each endpoint can be any of the above or
          ``None``. The slice operates on the underlying string (so ``step`` behaves
          like normal Python slicing).
        - **Tuples of length 2**: ``(key_or_slice, anything)`` returns the same
          selection but **annotated** with line numbers and starting character offsets.
          The second tuple element is ignored; its presence opts into line annotation.
        
        **Return modes**
        
        - Default: a substring of ``md`` for the requested character range.
        - Annotated mode (two-item tuple): a string with one line per visible line in
          the selection, each prefixed like ``"L12 : 1234 | <line text>"``, where:
          - ``L12`` is the absolute line number in the original document.
          - ``1234`` is the character offset where that printed line starts
            (for the first printed line, the exact slice start is used).
        
        :param keys: A dotted key, character offset, line key (``"L<n>"``), a slice of
                     any of those, or a 2-tuple to request the annotated view.
        :returns: Either the substring for the computed range or an annotated, line-
                  numbered view when a tuple was used.
        :raises IndexError: If a tuple of invalid arity is provided.
        :raises KeyError: If a non-existent dotted key or line key is provided.
        :raises ValueError: If a line key suffix cannot be parsed as an integer.
        
        **Examples**
        
        .. code-block:: python
        
            # Whole section by dotted key
            mdi["2.3"]
        
            # Character-span slice (first 100 chars)
            mdi[:100]
        
            # Lines 10 through 14 (inclusive of 10, exclusive of 15 as a char slice)
            mdi["L10":"L15"]
        
            # Section with line annotations
            print(mdi["2.3", None])
        
            # Cross-boundary slice using headings
            mdi["1.2":"1.4"]
        
            # Step slicing across characters (every other byte in a section)
            mdi["2.1":"2.2":2]
        """
        include_line_info = False
        #print(keys)
        if isinstance(keys, slice):
            start_key, end_key, step = keys.start, keys.stop, keys.step
        elif isinstance(keys, tuple):
            if len(keys)==2:
                key = keys[0] or keys[1]
                if isinstance(key, slice):
                    start_key, end_key, step =key.start, key.stop, key.step
                    include_line_info=True
                else:
                    start_key, end_key,step = keys[0],keys[0],None
                    include_line_info = True
            else:
                raise IndexError("Index argument mismatch.")
        else:
            start_key, end_key, step = keys, keys, None

        start_idx = self._resolve_key_to_index(start_key, True)
        end_idx = self._resolve_key_to_index(end_key, start_key!=end_key)

        substring = self.md[start_idx:end_idx:step]

        if include_line_info:
            start_idx = 0 if start_idx is None else start_idx
            start_line_idx = self._find_line_idx(start_idx)
            #print(start_line_idx)
            #end_line_idx = self._find_line_idx(end_idx)
            #isn= substring[0] is '\n' not it will never be
            lines = substring.split('\n')
            line_info = [f"L{i + start_line_idx} : {self.lindex[i + start_line_idx][0] if i>0 else start_idx} | {line}" for i, line in enumerate(lines)]
            #printed idx+1 so that counting is correct because newline is unseen in this printout.
            return '\n'.join(line_info)

        return substring

    def _resolve_key_to_index(self, key, is_start):
        if key is None:
            return None
        if isinstance(key, int):
            return key
        if key in self.index:
            return self.index[key][1] if is_start else self.index[key][2]

        # Added this block to handle line indexing

        if key[0] in ('l','L'):
            line_no = int(key[1:])
            if line_no < len(self.lindex):
                return self.lindex[line_no][0] if is_start else self.lindex[line_no][1]

        raise KeyError(f"Invalid key: {key}")

    def _find_line_idx(self, _idx):
        # Estimate the line index
        # Essentially a secant guess for discrete monotonic
        est_idx = int(_idx * (len(self.lindex) - 1) / len(self.md))
        # When searching for a line we look for the beginning of a new line.
        while self.lindex[est_idx][1] <= _idx:
            #print('up 1')
            est_idx += 1
        while self.lindex[est_idx][0] > _idx:
            #print('down 1')
            est_idx -= 1
        #Chose to while loop instead of bisect because any further secant guesses are likely to be innaccurate and we don't have a
        #POV for the near opposite bound, so bisection may have to proceed from the end or start of the markdown text.
        #Also the amount of loops required is often a tiny fraction of # lines for a normal dataset.
        return est_idx

    def idx_tuple(self, inx: str,tolast=True):
        """
        Convert a dotted section key into a fixed-length tuple suitable for
        range-comparisons across levels.
        
        The tuple length equals the number of heading levels represented in the
        document (``idx_max - idx_min + 1``). Missing deeper components are padded
        with zeros so that lexicographic comparison mirrors section ordering.
        
        Special handling is provided when ``inx`` is ``None``:
        
        - If ``tolast=False``, the first key in ``index`` is used.
        - If ``tolast=True`` (default), the last key in ``index`` is taken and the
          **first** component is incremented by one while all deeper components are set
          to zero. This produces a convenient exclusive upper bound sentinel for range
          scans (i.e., “just past the end” of the last top-level group).
        
        :param inx: A dotted key like ``"2.4.1"`` or ``None`` for sentinel behavior.
        :param tolast: Controls how ``None`` is resolved; see above.
        :returns: A tuple of integers suitable for ordering and boundary tests.
        
        **Examples**
        
        .. code-block:: python
        
            # Direct conversion
            mdi.idx_tuple("2.1.3")   # -> (2, 1, 3, 0, ...)
        
            # Lower/upper sentinels for slicing ranges
            lo = mdi.idx_tuple("1.2", tolast=False)
            hi = mdi.idx_tuple(None, tolast=True)   # one past the last top-level
        """
        spl =(inx if inx is not None else next(reversed(self.index.keys())) if tolast else next(iter(self.index.keys()))).split('.')
        lc = len(spl)
        if inx is not None or not tolast:
            return tuple(int(spl[i]) if i < lc else 0 for i in range(self.idx_max-self.idx_min+1))
        else:
            return tuple(int(spl[i])+1 if i < 1 else 0 for i in range(self.idx_max-self.idx_min+1))



def tuple_idx(tuple):
    return '.'.join([str(i) for i in tuple if i!=0])



def apply_filters(obj, *args):
    for f in args:
        obj = f(obj)
    return obj


def load_md(md_path, processors=(MDProcessors.redact_links,)):
    if processors is None:processors=()
    with open(md_path, 'r', encoding='utf-8') as mdf:
        return apply_filters(mdf.read(), *processors)

import anyio
async def aload_md(md_path, processors=(MDProcessors.redact_links,)):
    if processors is None:processors=()
    async with await anyio.open_file(md_path, 'r', encoding='utf-8') as mdf:
        return apply_filters(await mdf.read(), *processors)


_model_costs = {'gpt-4':(.03,.06),'gpt-3.5':(.0015,.002)} #per 1k tokens

def estimate_llmcost(mdi:MarkdownIndexer,*slices,model='gpt-4',input_ratio=1.+.33,output_ratio=.33, _tokenizer=GPTTOKEN): #guestimates on research reading.
    """
    Estimate the dollar cost of sending one or more selections of this markdown to
    an LLM, assuming a simple input/output token ratio model.
    
    The selections are specified using the **same addressing forms supported by
    ``__getitem__``** (single keys, slices, etc.). Each selection is extracted,
    tokenized with the configured tokenizer, and the token counts are summed to
    produce a cost estimate using the per-1k-token prices in ``_model_costs`` and
    the provided input/output ratios.
    
    :param mdi: A ``MarkdownIndexer`` instance to read from.
    :param slices: Zero or more selections (keys or slices). When omitted, the
                   entire document is used.
    :param model: A key present in ``_model_costs`` (e.g., ``"gpt-4"``). Determines
                  the per-1k token pricing for input and output.
    :param input_ratio: Multiplier applied to input token pricing. Use this to
                        approximate system/user prompt overhead.
    :param output_ratio: Multiplier applied to output token pricing. Use this to
                         budget for the model’s response length.
    :param _tokenizer: A callable with ``encode_ordinary_batch(List[str]) -> List[List[int]]``.
                       Provided to allow custom tokenizers in tests.
    :returns: A floating-point cost estimate in USD for the combined selections.
    :raises KeyError: If ``model`` is not present in ``_model_costs``.
    
    **Examples**
    
    .. code-block:: python
    
        # Whole document on gpt-4 with default ratios
        cost = estimate_llmcost(mdi)
    
        # Just chapter 2 and its appendices
        cost = estimate_llmcost(mdi, "2", "2.1":"2.4")
    
        # Tighter output budget
        cost = estimate_llmcost(mdi, "1":"3", output_ratio=0.15)
    """
    if len(slices)==0: slices=[slice(None,None)]
    tgroups = [len(tkg) for tkg in _tokenizer.encode_ordinary_batch([mdi[slic] for slic in slices])]
    ttks=sum(tgroups)

    return ttks*(_model_costs[model][0]*input_ratio + _model_costs[model][1]*output_ratio)/1000


def make_llmsections(mdi: MarkdownIndexer, *slices, mx_tokens=int(8192 * 5 / 8),isolate_section=False, _tokenizer=GPTTOKEN):
    """
    Partition the document into section-aligned chunks that do not exceed a target
    token budget, returning dotted keys, token totals, and exact character spans.
    
    This utility first expands the given selections into concrete **leaf-level
    sections** using the index (or uses all sections when no selections are
    supplied). It then tokenizes each section and groups **contiguous** sections
    into larger chunks so that the **sum of tokens per chunk** stays under
    ``mx_tokens`` while respecting natural boundaries in the heading hierarchy.
    
    Chunk boundaries are chosen to keep sections together when possible, split on
    higher-level transitions when needed, and always emit at least a full top-level
    section as its own chunk. The return value encodes both human-readable
    identifiers and precise character spans, so you can slice the original markdown
    directly.
    
    .. note::
       ``isolate_section=True`` is a placeholder and is not implemented; the
       function returns ``None`` in that case.
    
    :param mdi: A ``MarkdownIndexer`` instance to read from.
    :param slices: Zero or more addressers (keys or slices) to limit which sections
                  participate. Omitted means “all sections.”
    :param mx_tokens: Maximum number of tokens allowed per chunk.
    :param isolate_section: Reserved for future behavior that would forbid grouping
                            across siblings; currently not implemented.
    :param _tokenizer: A callable with ``encode_ordinary_batch(List[str])`` used to
                       count tokens per section.
    :returns: ``(chunk_keys, token_sums, char_spans)`` where:
              - ``chunk_keys`` is a list of dotted keys identifying the **first**
                section in each emitted chunk.
              - ``token_sums`` is a list of total tokens per chunk.
              - ``char_spans`` is a list of ``(start_char, end_char)`` for each
                chunk, suitable for slicing ``mdi.md[start:end]``.
    :raises ValueError: If no sections are available to group (e.g., empty index).
    
    **Examples**
    
    .. code-block:: python
    
        # Split the whole doc into ~1200-token pieces
        keys, sizes, spans = make_llmsections(mdi, mx_tokens=1200)
        parts = [mdi.md[s:e] for (s, e) in spans]
    
        # Only chapters 2–4
        keys, sizes, spans = make_llmsections(mdi, "2":"5", mx_tokens=2000)
    """
    # for now it only supports heading indexing, and no steps. if not slice it's a single value.
    # assuming sections are ordered and no duplicates.
    sec_idxs,sec_parts=_get_sectionidx(mdi,*slices)
    #print(sec_idxs)
    #print(sec_parts)
    if isolate_section:
        print('Isolated sections not implemented yet, use individual top layer selections for book chapters.')
        return None
    else:
        tgroups=[len(tkg) for tkg in _tokenizer.encode_ordinary_batch([mdi.md[s0:s1] for s0,s1 in sec_parts])]
        #print(tgroups)
        idx_tples,sums,exact_idxs = group_layers(sec_idxs,tgroups,sec_parts,mx_tokens)
        #print(idx_tples,sums,exact_idxs)
        return [tuple_idx(i) for i in idx_tples],sums,exact_idxs



def _get_sectionidx(mdi: MarkdownIndexer, *slices):
    
    sectns = []
    secgps = []
    if len(slices) == 0 or slices[0] is None:
        sectns.extend(map(mdi.idx_tuple, mdi.index.keys()))
        return [*map(mdi.idx_tuple, mdi.index.keys())],[vl[1:] for vl in mdi.index.values()]
    else:
        sli = iter(slices)
        sec = next(sli, None)
        sec = mdi.idx_tuple(sec) if not isinstance(sec, slice) else (
        mdi.idx_tuple(sec.start, tolast=False), mdi.idx_tuple(sec.stop, tolast=True))

    for i, kv in enumerate(mdi.index.items()):
        key = kv[0]
        grp = kv[1][1:]
        kid = mdi.idx_tuple(key)
        if type(sec[0]) is tuple:
            if kid < sec[0]:
                continue
            elif kid >= sec[1]:
                sec = next(sli, None)
                if sec is None: break
                sec = mdi.idx_tuple(sec) if not isinstance(sec, slice) else (
                mdi.idx_tuple(sec.start, tolast=False), mdi.idx_tuple(sec.stop, tolast=True))
            else:
                sectns.append(kid)
                secgps.append(grp)
                continue
        else:
            # sec is a singular value
            if kid < sec:
                continue
            elif kid > sec:
                sec = next(sli, None)
                if sec is None: break
                sec = mdi.idx_tuple(sec) if not isinstance(sec, slice) else mdi.idx_tuple(sec.stop, tolast=False), mdi.idx_tuple(
                    sec.end, tolast=True)
            else:
                sectns.append(kid)
                secgps.append(grp)
                continue
        # if not continued then try the next sec but without the need of finding the next one
        if type(sec[0]) is tuple:
            if sec[0] <= kid < sec[1]:
                sectns.append(kid)
                secgps.append(grp)
        elif kid == sec:
            sectns.append(kid)
            secgps.append(grp)
    return sectns, secgps

import numpy as np


def group_layers(tuples_list, int_list,idx_list, mx):
    """
    Greedy, hierarchy-aware grouper for section tuples and sizes.
    
    Given:
    
    - ``tuples_list`` — the section identifiers as **fixed-length tuples** (e.g.,
      ``(2, 1, 0, 0)``) ordered exactly as they appear in the document,
    - ``int_list`` — the size for each section (e.g., token counts),
    - ``idx_list`` — the exact ``(start_char, end_char)`` span for each section,
    
    this function emits chunk boundaries that:
    
    1. Keep adjacent sections together until the running total would exceed ``mx``.
    2. Prefer to split on **higher-level** transitions (chapter → next chapter)
       before lower ones when possible.
    3. Always include **top-level** sections as independent candidates so that a
       whole chapter is never silently merged into the next.
    
    The result is three parallel lists:
    
    - ``result_tuples`` — the tuple key of the **first** section in each emitted
      chunk (these align with dotted keys via ``tuple_idx`` elsewhere).
    - ``result_sums`` — the sum of sizes within each chunk.
    - ``result_idxs`` — the merged character span of all sections in the chunk,
      suitable for slicing the source text.
    
    Internally, the algorithm tracks running sums per layer and resets/flushes
    groups whenever a tuple’s prefix changes at that layer. When a layer’s running
    sum exceeds ``mx``, the current group at that layer is closed and emitted.
    
    :param tuples_list: Ordered list of section tuple IDs (all the same depth).
    :param int_list: Parallel list of integer sizes for each section.
    :param idx_list: Parallel list of ``(start_char, end_char)`` spans per section.
    :param mx: Maximum allowed sum per emitted group.
    :returns: ``(result_tuples, result_sums, result_idxs)`` as described above.
    :raises AssertionError: If the three input lists differ in length.
    
    **Examples**
    
    .. code-block:: python
    
        # Suppose chapters 1..3 with small subsections
        tuples = [(1,0), (1,1), (1,2), (2,0), (2,1), (3,0)]
        sizes  = [800,   300,   400,   900,   500,   700]
        spans  = [(0,100),(100,150),(150,220),(220,330),(330,400),(400,480)]
    
        # With mx=1000, this will prefer cuts at top-level boundaries,
        # but will also split mid-chapter if needed.
        tkeys, sums, idxs = group_layers(tuples, sizes, spans, mx=1000)
    """
    # Initialize variables
    depth = len(tuples_list[0])  # Depth of the tuples
    layer_sum, layer_idx = np.zeros(depth-1,dtype=np.int32), np.zeros(depth-1,dtype=np.int32)  # Counters for each layer
    lyr_grps=[[] for _ in range(depth-1)]
    new_group = np.zeros(depth-1,dtype=bool)
    result_tuples, result_sums,result_idxs = set(), [],[]  # Result lists
    #can do a two pass, add up results sum afterwards easier I think.

    # Iterate through the list of tuples and integer values
    for p in range(len(int_list)):
        id_t=tuples_list[p]
        sz=int_list[p]
        new_group[:]=np.maximum.accumulate(layer_idx<id_t[:-1])
        #print(new_group)
        #only check on/after layer switch
        #adding and incrementing comes after resets.
        for i in range(0,depth-1):
            if new_group[i]:
                if layer_sum[i]>mx:
                    result_tuples.update(lyr_grps[i])
                layer_sum[i]=0
                layer_idx[i:]=id_t[i:-1]
                lyr_grps[i].clear()
                lyr_grps[i].append(id_t)
            elif i+1==depth or sum(id_t[i+2:])==0: #see if indexing logic can be made more clean
                lyr_grps[i].append(id_t)
        layer_sum+=sz
        # layer 1 get's special treatment
        if sum(id_t[1:]) == 0:
            result_tuples.add(id_t)

    result_tuples=[r for r in result_tuples]
    lnt=len(tuples_list)
    result_tuples.sort()
    result_tuples.append((result_tuples[-1][0]+1,))
    idx = 0
    for tpl in result_tuples[1:]:
        csz = int_list[idx]
        ist=idx_list[idx][0]
        idx+=1
        if idx==lnt or tuples_list[idx]==tpl:
            result_sums.append(csz)
            result_idxs.append((ist,idx_list[idx-1][1]))
        else:
            cmp=True
            while True:
                idx += 1
                if idx==lnt or tuples_list[idx] >= tpl:
                    break
                csz += int_list[idx]
            result_sums.append(csz)
            result_idxs.append((ist,idx_list[idx-1][1]))
    result_tuples.pop(-1)

    return result_tuples, result_sums,result_idxs
