# Logistic Regression

It is a type of supervised learning in which we are given some features using them we have to classify them into one of many categories. Example, finding type of wine (good or bad) (target value) given the pH, acidity etc (features).

### Binary Logistic Regression

Let our training set be $\mathcal{D} = \{(x_1,y_1),(x_2,y_2) \cdots (x_m,y_m)\}$ where $x_m \in \R^n$ are the features and $y_m \in \{0,1\}$ are our target values i.e. there are only two categories.

<aside>
📌 As there is a possibility that different input features have extremely different ranges of values, hence the columns are generally normalised i.e. standardising the input values by centring them to result in a zero mean and a standard deviation of one.

</aside>

With the given training inputs $x_i \in \R^n$ and corresponding observations $y_i \in \R$ we need to find $p(y =1 \mid x)$. Like linear regression, each column is given a weight i.e. 

$$
z_i = x_i^T\theta
$$

where $\theta\in \R^{n}$ are our parameters, but here the value $z_i \in(-\infin, \infin)$ . So we define a sigmoid function $\sigma(z)$ such that $\sigma : (-\infin,\infin) \to (0,1)$ i.e. 

$$
\sigma(z) = {1 \over 1+ \exp(-z)}
$$

Hence  

$$
p(y=1\mid x) =\sigma(x_i^T\theta)
$$

So our predicted values $y_i^*$ is defined as following 

$$
y_i^* = \begin{cases}1 & p(y=1\mid x) > 0.5\\ 0 &\ce{otherwise}\end{cases}
$$

<aside>
📌 In predicting values we can add another parameter, the bias i.e. now $z_i=x_i^T\theta + b$ where $b$ is our bias. 
Instead of above we can also add a column of ones to the data set. So one of the parameter in $\theta$ will act as the replacement for $b$. The only difference in below calculations will be the dimension will change from $n \to n+1$

</aside>

So we need to maximise $p(y\mid x)$ $\forall$ inputs $x_i$ and features $y_i$ which is same as maximising 

$$
\prod_{i=1}^m p(y_i \mid x_i)
$$

As $p(y\mid x)$ has only two target values possible, hence this is a Bernoulli distribution i.e.  

$$
p(y\mid x) = \hat{y}^y(1-\hat{y})^{1-y}
$$

where $\hat{y}$ is $p(y=1 \mid x)$

Defining a loss function 

$$
\begin{align*}\mathcal{L}(\theta) &= -\log p(y_i\mid x_i)\\ &= -y_i\log\hat{y}_i - (1-y_i)\log(1-\hat{y}_i)\\&=-y_i\log(\sigma(x_i^T\theta))-(1-y_i)\log(1-\sigma(x^T_i\theta))\end{align*}
$$

So we have to minimise the loss function

$$
\begin{align*}{d\mathcal{L} \over d\theta}&= -{y_i \over \sigma(x_i^T\theta)}{d\over d\theta}(\sigma(x_i^T\theta)) -{1-y \over 1-\sigma(x^T\theta)}{d\over d\theta}(1-\sigma(x_i^T\theta))\\ &=- \bigg[{y_i \over \sigma(x_i^T\theta)}  - {1-y \over 1-\sigma(x^T\theta)}\bigg]{d\over d\theta}(\sigma(x_i^T\theta)) \\&=-\bigg({y_i-\sigma(x^T_i\theta)\over \sigma(x_i^T\theta)(1-\sigma(x_i^T\theta))}\bigg)\sigma(x_i^T\theta)(1-\sigma(x_i^T\theta))x_i\\ &=(\sigma(x^T_i\theta)-y_i)x_i \end{align*}
$$

Now we can apply gradient descent where $\displaystyle \delta\theta = {d\mathcal{L} \over d\theta}$ i.e. 

$$
\theta(t) := \theta(t-1) - \eta {d\mathcal{L} \over d\theta}\bigg|_{\theta(t-1)}
$$

where $\eta >0$ is our learning rate. We are going in the opposite direction of gradient which helps in getting closer towards the minimum. 

<aside>
📌 We can also make the derivative of the loss function directly $0$ but it will be hard to calculate the value of $\theta$ directly this way that’s why help of gradient descent is taken.

</aside>

### Multinomial Logistic Regression

Now $y_i \in \{1,2,\cdots,k\}$. Now we will predict a vector $y_i^* \in \R^k$ for each row such that  

$$
y_i^* = \begin{bmatrix}p(y_i=1\mid x_i),p(y_i=2\mid x_i),\cdots,p(y_i=k\mid x_i)\end{bmatrix}
$$

The one with maximum probability will be the category of that row

With the given training inputs $x_i \in \R^n$ and corresponding observations $y_i \in \R^k$ we need to find $y_i^*$ 

$$
z_i = [z_{i1},z_{i2},\cdots,z_{ik}] = [x_i^T\theta_1,x_i^T\theta_2,\cdots ,x_i^T\theta_k]
$$

where $\theta_j\in \R^{n}$ is the parameter. It can also be redefined as the following 

$$
z_i = x_i^T\Theta
$$

where $\Theta \in \R^{n \times k}$ such that $\Theta = [\theta_1^T,\theta_2^T,\cdots,\theta_k^T]^T$ 

Defining softmax function like sigmoid such that 

$$
y_i^*=\ce{softmax}(z_i) =\begin{bmatrix} \frac{\exp(z_{i1})}{ \sum_{j=1}^k \exp(z_{ij})},\frac{\exp(z_{i2})}{ \sum_{j=1}^k \exp(z_{ij})},\cdots,\frac{\exp(z_{ik})}{ \sum_{j=1}^k \exp(z_{ij})} \end{bmatrix}
$$

The maximum value of the array $y_i^*$ will be our class.

Here the probability distribution will be 

$$
p(y_i \mid x_i) = \prod_{j=1}^k\hat{y_{ij}}^{y_{ij}}
$$

where $\hat{y_{ij}} = p(y_i = j \mid x_i) = z_{ij}$ and $y_{i1},y_{i2}\cdots y_{ik}$ are such that only one of them will be $1$ and rest of them will be $0$ i.e. $p(y_i=j \mid x_i)$ will have $y_{il} = 0 \forall l\ne j$ and $y_{ij} = 1$.

Defining the loss function 

$$
\begin{align*}\mathcal{L}(\Theta) &= -\log p(y_i\mid x_i)\\ &= -\sum_{j=1}^k y_{ij}\log\hat{y}_{ij}\\ &= -\sum_{j=1}^ky_{ij}\big[\log(e^{z_{ij}})-\log\big(\sum_{j=1}^k e^{z_{ij}}\big)\big]\\ &= -\sum_{j=1}^ky_{ij}(z_{ij}-\log\big(\sum_{j=1}^k e^{z_{ij}}\big))\\ &= -\sum_{j=1}^ky_{ij}(x_i\theta_j - \log\big(\sum_{j=1}^k e^{z_{ij}}\big))\\&=-\sum_{j=1}^k y_{ij}(x_i^T\theta_j) + \log\big(\sum_{j=1}^k e^{z_{ij}}\big)\sum_{j=1}^ky_{ij}\\ &= -\sum_{j=1}^k y_{ij}(x_i^T\theta_j) + \log\big(\sum_{j=1}^k e^{z_{ij}}\big) \end{align*}
$$

 Finding the gradient of the loss function 

$$
\nabla \mathcal{L}(\Theta) = \begin{bmatrix}{\partial \mathcal{L} \over \partial\theta_1}\\ {\partial \mathcal{L} \over \partial\theta_2} \\ \vdots \\ {\partial \mathcal{L} \over \partial\theta_k}\end{bmatrix}
$$

$$
\begin{align*}{\partial \mathcal{L} \over \partial \theta_j} &= -y_{ij}x_i^T + {e^{x_i^T\theta_j}x_i^T \over \sum_{j=1}^ke^{x_i^T\theta_j}}\\ &= (y^*_{ij} - y_{ij})x_i^T \end{align*}
$$

where $y_{ij}^* =\displaystyle {e^{x_i^T\theta_j}x_i^T \over \sum_{j=1}^ke^{x_i^T\theta_j}}$ which is a similar result as in binomial logistic regression.

Now we can again apply gradient descent  and find the  best fitting parameters

### Questions

1. **What is a Bernoulli distribution ?**
    
    A random variable will have this distribution if it has only two outcomes. Let the outcomes be $\{0,1\}$ then the distribution will be 
    
    $$
    p(y\mid x) = \hat{y}^y(1-\hat{y})^{(1-y)}
    $$
    
    where $\hat{y} = p(y=1\mid x)$. So when $y = 1$ is substituted in above equation, it will result in 
    
    $$
    p(y=1\mid x) = \hat{y}
    $$
    
    when $y =0$ is substituted in above equation, it will result in 
    
    $$
    p(y=0\mid x) = 1-\hat{y}
    $$
    
2. **In gradient descent method for finding the best parameters how will we adjust the learning rate i.e. $\eta$ ?**
    
    We can find the best learning rate using cross validation. 
    
3. **Why to use gradient descent instead of directly making the derivative $0$ ?**
    
     Making the derivative directly equal to $0$ takes a lot of computational power as compared to the gradient descent. Moreover, in large data set the exact derivative and the approximate minimum found using graident descent doesn’t make a significant difference.
    
4. **Logistic regression distributes a data into different categories but its still called as regression instead of classification why ?**
    
    Logistic regression is called regression because it finds out the approximate probability of a data point to belong to a class which is an exact value. Hence it is a regression and not a classification. 
    
5. **Why is sigmoid or softmax function used ?**
    
    To make the range $(0,1)$ we use sigmoid or softmax function.