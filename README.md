# Markdown Parser

Refers to text blocks through multiple levels of abstraction, starting at headers, all the way down to lines and individual character offsets.
Also includes a hierarchical text segmentation algorithm to fit a token constraint, largest headers prioritized first. Has utility
in RAG, embeddings, and MPC.

See [demo](demo.md) for usage.

The `MarkdownIndexer` class will construct an explicitly-sectioned outline of the text through the headers.  
A line starting with the least number of `#` characters will set the top level section, sections with more `#`'s will receive an
extra '.'

    1 : ('Global Convergence of Online Limited Memory BFGS', 0, 277)
    1.0.0.1 : ('Abstract', 277, 1065)
    1.1 : ('Introduction', 1065, 12001)
    1.2 : ('Algorithm Definition', 12001, 17592)
    1.2.1 : ('LBFGS: Limited Memory BFGS', 17592, 21426)
    1.2.2 : ('Online (Stochastic) Limited Memory BFGS', 21426, 39693)
    1.3 : ('Convergence Analysis', 39693, 60907)
    1.4 : ('Search Engine Advertising', 60907, 62267)
    1.4.1 : ('Feature Vectors', 62267, 68797)
    1.4.2 : ('Logistic Regression of Click-Through Rate', 68797, 72733)
    1.4.3 : ('Numerical Results', 72733, 86774)
    1.5 : ('Conclusions', 86774, 87685)
    1.6 : ('Acknowledgments', 87685, 87850)
    1.7 : ('Appendix A. Proof of Proposition 1', 87850, 95394)
    1.8 : ('Appendix B. Proof of Lemma 2', 95394, 99527)
    1.9 : ('Appendix C. Proof of Lemma 3', 99527, 113740)
    1.10 : ('Appendix D. Proof of Lemma 4', 113740, 116194)
    1.11 : ('Appendix E. Proof of Lemma 5', 116194, 120723)
    1.12 : ('Appendix F. Proof of Theorem 6', 120723, 127770)
    1.13 : ('Appendix G. Proof of Theorem 7', 127770, 136021)
    1.14 : ('References', 136021, 137372)
