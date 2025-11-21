# Markdown Parser

Refers to text blocks through multiple levels of abstraction, starting at headers, all the way down to lines and individual character offsets.
Also includes a hierarchical text segmentation algorithm to fit a token constraint, largest headers prioritized first. Has utility
in RAG, embeddings, and MPC.

See [demo](demo.md) for usage.

The `MarkdownIndexer` class will construct an explicitly-sectioned outline of the text through the headers.  
A line starting with the least number of `#` characters will set the top level section, sections with more `#`'s will receive an
extra '.'

    1 : Global Convergence of Online Limited Memory BFGS
             1.0.0.1 : Abstract
       1.1 : Introduction
       1.2 : Algorithm Definition
          1.2.1 : LBFGS: Limited Memory BFGS
          1.2.2 : Online (Stochastic) Limited Memory BFGS
       1.3 : Convergence Analysis
       1.4 : Search Engine Advertising
          1.4.1 : Feature Vectors
          1.4.2 : Logistic Regression of Click-Through Rate
          1.4.3 : Numerical Results
       1.5 : Conclusions
       1.6 : Acknowledgments
       1.7 : Appendix A. Proof of Proposition 1
       1.8 : Appendix B. Proof of Lemma 2
       1.9 : Appendix C. Proof of Lemma 3
       1.10 : Appendix D. Proof of Lemma 4
       1.11 : Appendix E. Proof of Lemma 5
       1.12 : Appendix F. Proof of Theorem 6
       1.13 : Appendix G. Proof of Theorem 7
       1.14 : References
    Markdown is 1020 lines long, with total character length 137372.
