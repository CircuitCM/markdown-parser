import re
from collections import OrderedDict
from typing import Iterable, Collection
from functools import cached_property
from itertools import islice
import tiktoken

GPTTOKEN= tiktoken.get_encoding("cl100k_base")


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
        '''order_sections=True : places the possibly unstructured/loosely structured Markdown headers with the structured indexing of the class.
        Because the class indexing won't actually correspond to the numbers in the header if they aren't counted properly.
        shift_headings=True : Shifts all markdown headings down by it's minimum level. From hn - hn+m to h1 - h1+m
        If path is None, simply returns a new MarkdownIndexer with the changes, doesn't write to file also.'''
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
                    new_header = f"{'#' * new_header_level}"
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

    def __getitem__(self, keys):
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

        # This is gotta be more efficient than a binary search.

    def _find_line_idx(self, _idx):
        # Estimate the line index
        est_idx = int(_idx * (len(self.lindex) - 1) / len(self.md))
        # When searching for a line we look for the beginning of a new line.
        while self.lindex[est_idx][1] <= _idx:
            #print('up 1')
            est_idx += 1
        while self.lindex[est_idx][0] > _idx:
            #print('down 1')
            est_idx -= 1
        return est_idx

    def idx_tuple(self, inx: str,tolast=True):
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
    with open(md_path, 'r', encoding='utf-8') as mdf:
        return apply_filters(mdf.read(), *processors)

import anyio
async def aload_md(md_path, processors=(MDProcessors.redact_links,)):
    async with await anyio.open_file(md_path, 'r', encoding='utf-8') as mdf:
        return apply_filters(await mdf.read(), *processors)


_model_costs = {'gpt-4':(.03,.06),'gpt-3.5':(.0015,.002)} #per 1k tokens

def estimate_llmcost(mdi:MarkdownIndexer,*slices,model='gpt-4',input_ratio=1.+.33,output_ratio=.33, _tokenizer=GPTTOKEN): #guestimates on research reading.
    if len(slices)==0: slices=[slice(None,None)]
    tgroups = [len(tkg) for tkg in _tokenizer.encode_ordinary_batch([mdi[slic] for slic in slices])]
    ttks=sum(tgroups)

    return ttks*(_model_costs[model][0]*input_ratio + _model_costs[model][1]*output_ratio)/1000


def make_llmsections(mdi: MarkdownIndexer, *slices, mx_tokens=int(8192 * 5 / 8),isolate_section=False, _tokenizer=GPTTOKEN):
    # for now it only supports heading indexing, and no steps. if not slice it's a single value.
    # assuming sections are ordered and no duplicates.
    sec_idxs,sec_parts=_get_sectionidx(mdi,*slices)
    print(sec_idxs)
    print(sec_parts)
    if isolate_section:
        print('Isolated sections not implemented yet, use individual top layer selections for book chapters.')
        return None
    else:
        tgroups=[len(tkg) for tkg in _tokenizer.encode_ordinary_batch([mdi.md[s0:s1] for s0,s1 in sec_parts])]
        print(tgroups)
        idx_tples,sums,exact_idxs = group_layers(sec_idxs,tgroups,sec_parts,mx_tokens)
        print(idx_tples,sums,exact_idxs)
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


