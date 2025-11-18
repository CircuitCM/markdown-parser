```python
from parse import MarkdownIndexer,MDProcessors,load_md
import os

txt=load_md(r'D:\OneDrive - The University of Chicago\Research\Useful\Gradient Optimizer Research\oLBFGS\ocr\oLBFGS_copy.md',())#(MDProcessors.redact_links,))

md = MarkdownIndexer(txt)
md #To display the section index and summary info
```




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




```python
md[:] #return the entire text.
```


```python
print('\n'.join(f"{k} : {v}" for k, v in md.index.items()))
```

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
    


```python
print(md['1.0.0.1':'1.1', None]) #for line and character offset info
#Note that we have to fully type '1.0.0.1', this it to avoid the ambiguity of referencing '1' or '1.0'
```

    L7 : 277 | #### Abstract
    L8 : 291 | 
    L9 : 292 | Global convergence of an online (stochastic) limited memory version of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method for solving optimization problems with stochastic objectives that arise in large scale machine learning is established. Lower and upper bounds on the Hessian eigenvalues of the sample functions are shown to suffice to guarantee that the curvature approximation matrices have bounded determinants and traces, which, in turn, permits establishing convergence to optimal arguments with probability 1. Experimental evaluation on a search engine advertising problem showcase reductions in convergence time relative to stochastic gradient descent algorithms.
    L10 : 980 | 
    L11 : 981 | 
    L12 : 982 | Keywords: quasi-Newton methods, large-scale optimization, stochastic optimization
    L13 : 1064 | 
    L14 : 1065 | 
    


```python
print(md['1.0.0.1':'1.1']) #Just the text
```

    #### Abstract
    
    Global convergence of an online (stochastic) limited memory version of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method for solving optimization problems with stochastic objectives that arise in large scale machine learning is established. Lower and upper bounds on the Hessian eigenvalues of the sample functions are shown to suffice to guarantee that the curvature approximation matrices have bounded determinants and traces, which, in turn, permits establishing convergence to optimal arguments with probability 1. Experimental evaluation on a search engine advertising problem showcase reductions in convergence time relative to stochastic gradient descent algorithms.
    
    
    Keywords: quasi-Newton methods, large-scale optimization, stochastic optimization
    
    
    


```python
print(md['L10':'L12',None]) #Just specific lines
print('---')
print(md['L10':'L13',None]) 
#Showing an empty last line is the natural way the offset of the final line can be represented so we leave it in the output.
```

    L10 : 980 | 
    L11 : 981 | 
    L12 : 982 | 
    ---
    L10 : 980 | 
    L11 : 981 | 
    L12 : 982 | Keywords: quasi-Newton methods, large-scale optimization, stochastic optimization
    L13 : 1064 | 
    


```python
print(md[297:500,None])  #Just characters numbers, note that the character index no longer corresponds to the start of the line.
```

    L9 : 297 | l convergence of an online (stochastic) limited memory version of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method for solving optimization problems with stochastic objectives that arise i
    


```python
print(md['L9':400,None]) #line to character
print('---')
print(md[276:'L9',None])
```

    L9 : 292 | Global convergence of an online (stochastic) limited memory version of the Broyden-Fletcher-Goldfarb-Shanno 
    ---
    L6 : 276 | 
    L7 : 277 | #### Abstract
    L8 : 291 | 
    L9 : 292 | 
    


```python
print(md['1':256,None]) #section to character
```

    L0 : 0 | # Global Convergence of Online Limited Memory BFGS 
    L1 : 52 | 
    L2 : 53 | Aryan Mokhtari<br>Alejandro Ribeiro<br>Department of Electrical and Systems Engineering<br>University of Pennsylvania<br>Philadelphia, PA 19104, USA<br>ARYANM@SEAS.UPENN.EDU<br>ARIBEIRO@SEAS.UPENN.EDU
    L3 : 254 | 
    L4 : 255 | E
    

## Reformatting Markdown




```python
mdt=MarkdownIndexer('#'+md[:])
mdt
```




    1 : Global Convergence of Online Limited Memory BFGS
          1.0.1 : Abstract
    2 : Introduction
    3 : Algorithm Definition
       3.1 : LBFGS: Limited Memory BFGS
       3.2 : Online (Stochastic) Limited Memory BFGS
    4 : Convergence Analysis
    5 : Search Engine Advertising
       5.1 : Feature Vectors
       5.2 : Logistic Regression of Click-Through Rate
       5.3 : Numerical Results
    6 : Conclusions
    7 : Acknowledgments
    8 : Appendix A. Proof of Proposition 1
    9 : Appendix B. Proof of Lemma 2
    10 : Appendix C. Proof of Lemma 3
    11 : Appendix D. Proof of Lemma 4
    12 : Appendix E. Proof of Lemma 5
    13 : Appendix F. Proof of Theorem 6
    14 : Appendix G. Proof of Theorem 7
    15 : References
    Markdown is 1020 lines long, with total character length 137373.



Despite outer alignment of the section indexes we can see that the largest header is now '##':


```python
print(mdt['2':1085,None]) #Note how the actual header number doesn't match the markdown outline either.
```

    L14 : 1066 | ## 1. Introduction
    L15 : 1085 | 
    

 But we can alter this so that the outermost heading is always '#' and explicitly add ordering:


```python
#The path is optional, setting it to None will return the reformatted text and new markdown object.
mdt2=mdt.fix_write(r'D:\OneDrive - The University of Chicago\Research\Useful\Gradient Optimizer Research\oLBFGS\ocr\oLBFGS_copy2.md',order_sections=True,shift_headings=True)
print(mdt2['2':'L15',None])
```

    L14 : 1074 | # 2. Introduction
    L15 : 1092 | 
    
