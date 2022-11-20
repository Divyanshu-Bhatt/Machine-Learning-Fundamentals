# Naive Bayes

It is a type of supervised learning algorithm which usually helps in classifying a data point into different categories. One of the major use of Naive Bayes is in mail spam filter.

It is a generative algorithm unlike linear or logistic regression which are discrimative algorithms. A generative algorithm finds both $p(x \mid y)$ and $p(y)$ instead of $p(y\mid x)$ where $y$ are our target values and $x$ are our features and using Bayes theorem it finds $p(y \mid x)$. But $p(x)$ is constant hence instead  $p(y\mid x)$ we find $p(x,y)$ because

$$
p(y\mid x) = {p(x\mid y) p(y) \over p(x)}\\
p(x,y) = p(x\mid y) p(y)
$$

 

Lets take the example of spam filter. First we have to map the text of the mail to a vector. This can be done by checking whether a specific word is in the text or not. 

So our vector $x_i \in\R^k$ (the $i^{th}$ mail) will contain $0$s and $1$s. Each index of the vector $x_i$ will correspond to a word and value $1$ at that index means that, that word is present in the text otherwise it would be $0$.  So

$$
x_{ij} = \mathcal{I}\{\ce{if word corresponding to the j^{th} index is present in the text\}}
$$

where $\mathcal{I}\{\}$ returns $1$ if the statement inside it is correct otherwise it returns $0$. Let we define $X_1,X_2,\cdots X_k$ are the all different words and $X_i$ corresponds to the $i^{th}$ index

Moreover, we consider that $x_{ij}$ are conditionally independent given $y$ i.e. 

$$
p(x_{i1},x_{i2},\cdots,x_{ik}\mid y_i) = \prod_{j=1}^kp(x_{ij}\mid y)
$$

So now are data set is $\mathcal{D} = \{(x_1,y_1),\cdots,(x_m,y_m)\}$ where we created the vectors $x_1 \cdots x_m$ for each mail and $y_i\in \{0,1\}$ i.e. $y_i=0$ means it is not a spam mail $y_i=1$ means it is a spam mail

We need to find $\phi_{j\mid y=1}=p(X_j\mid y=1)$ i.e. the probability if the $j^{th}$ word is present in the text if its a spam mail and similarly $\phi_{j\mid y=0}=p(X_j \mid y=0)$ and  $\phi_{y}=p(y=1)$ i.e. the probability that it is a spam mail.

With the given data it can be find by  

$$
\phi_{j\mid y=1} = {\displaystyle\sum_{i=1}^m \mathcal{I}\{x_{ij}=1,y_i=1\}\over \displaystyle\sum_{i=1}^m\mathcal{I}\{y_i=1\}}
$$

$$
\phi_{j\mid y=0} = {\displaystyle\sum_{i=1}^m \mathcal{I}\{x_{ij}=1,y_i=0\}\over \displaystyle\sum_{i=1}^m\mathcal{I}\{y_i=0\}}
$$

$$
\phi_{y} = {\displaystyle\sum_{i=1}^m \mathcal{I}\{y_i=1\}\over m}
$$

So $\phi_{y=1},\phi_{y=0}\in \R^k$ (all the different words) 

So for a new mail we defined the vector $z$  

$$
\begin{align*}p(y=1\mid z) & = {p(z\mid y=1) p(y=1) \over p(z\mid y=0) p(y=0)+p(z\mid y=1) p(y=1)}\\\end{align*} 
$$

The problem with the above method is that if a new word arise in the test set that was never seen before in the training set then $p(z\mid y=1)=p(z\mid y=0) = 0$ which will result in the above probability as $0\over 0$. Hence we use Laplace smoothing.

For each outcome possible we add a $1$ i.e. 

$$
\phi_{j\mid y=1} = {\displaystyle\sum_{i=1}^m \mathcal{I}\{x_{ij}=1,y_i=1)+1\over \displaystyle\sum_{i=1}^m\mathcal{I}\{y_i=1\}+2}
$$

$$
\phi_{j\mid y=0} = {\displaystyle\sum_{i=1}^m \mathcal{I}\{x_{ij}=1,y_i=0\}+1\over \displaystyle\sum_{i=1}^m\mathcal{I}\{y_i=0\}+2}
$$

### Questions

1. **What is the difference between generative and discriminative algorithm ?**
    
    A generative model focuses on explaining how the data was generated example naive bayes algorithm, while a discriminative model focuses on predicting the labels of the data example logistic regression.
    
2. **What is Bayes Theorem ?**
    
    $$
    \underbrace{p(x\mid y)}_\text{posterior} ={\overbrace{p(y\mid x)}^\text{likelihood}\overbrace{p(x)}^\text{prior}\over \underbrace{p(y)}_\text{evidence}}
    $$
    
3. **Why is Naive Bayes algorithm is called Naive ?**
    
    It is called Naive because it doesn’t depend on the order of the words. Example the probability for “Respected Sir” and “Sir Respected” would be the same. Moreover it ignores the grammatical rules
    
4. **What is the main assumption we take in Naive Bayes algorithm ?**
    
    The main assumption of Naive Bayes algorithm is all data are conditionally independent i.e. 
    
    $$
    p(x_1,x_2\mid y) = p(x_1\mid y)p(x_2\mid y)
    $$
    
5. **Why do we need Laplace Smoothing ?**
    
    To avoid the case of ${0 \over 0}$ when a new word comes in the test set which was never seen before in the training set.