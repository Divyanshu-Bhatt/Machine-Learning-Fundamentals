# Logistic Regression

Logistic Regression is a type of supervised learning in which we are given some features using them we have to classify them into one of many categories. Example, finding type of wine (good or bad) (target value) given the pH, acidity etc (features).

### Binary Logistic Regression

Let our training set be $\mathcal{D} = \{(x_1,y_1),(x_2,y_2) \cdots (x_m,y_m)\}$ where $x_m \in \mathbb{R}^n$ are the features and $y_m \in \{0,1\}$ are our target values i.e. there are only two categories.

With the given training inputs $x_i \in \mathbb{R}^n$ and corresponding observations $y_i \in \mathbb{R}$ we need to find $p(y =1 \mid x)$. Like linear regression, each column is given a weight i.e. 

$$
z_i = x_i^T\theta
$$

where $\theta\in \mathbb{R}^{n}$ are our parameters, but here the value $z_i \in(-\infty, \infty)$ . So we define a sigmoid function $\sigma(z)$ such that $\sigma : (-\infty,\infty) \to (0,1)$ i.e. 

$$
\sigma(z) = {1 \over 1+ \exp(-z)}
$$

Hence  

$$
p(y=1\mid x) =\sigma(x_i^T\theta)
$$

So our predicted values $y_i^*$ is defined as following 

$$
y_i^* = \begin{cases}1 & p(y=1\mid x) > 0.5\newline 0 &\ce{otherwise}\end{cases}
$$

In predicting values we can add another parameter, the bias i.e. now $z_i=x_i^T\theta + b$ where $b$ is our bias. 
Instead of above we can also add a column of ones to the data set. So one of the parameter in $\theta$ will act as the replacement for $b$. The only difference in below calculations will be the dimension will change from $n \to n+1$

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
\begin{align}\mathcal{L}(\theta) &= -\log p(y_i\mid x_i) \newline &= -y_i\log\hat{y}_i - (1-y_i)\log(1-\hat{y}_i)\newline &=-y_i\log(\sigma(x_i^T\theta))-(1-y_i)\log(1-\sigma(x^T_i\theta))\end{align}
$$

So we have to minimise the loss function

$$
\begin{align}{d\mathcal{L} \over d\theta}&= -{y_i \over \sigma(x_i^T\theta)}{d\over d\theta}(\sigma(x_i^T\theta)) -{1-y \over 1-\sigma(x^T\theta)}{d\over d\theta}(1-\sigma(x_i^T\theta))\newline &=- \bigg[{y_i \over \sigma(x_i^T\theta)}  - {1-y \over 1-\sigma(x^T\theta)}\bigg]{d\over d\theta}(\sigma(x_i^T\theta)) \newline &=-\bigg({y_i-\sigma(x^T_i\theta)\over \sigma(x_i^T\theta)(1-\sigma(x_i^T\theta))}\bigg)\sigma(x_i^T\theta)(1-\sigma(x_i^T\theta))x_i\newline &=(\sigma(x^T_i\theta)-y_i)x_i \end{align}
$$

Now we can apply gradient descent where $\displaystyle \delta\theta = {d\mathcal{L} \over d\theta}$ i.e. 

$$
\theta(t) := \theta(t-1) - \eta {d\mathcal{L} \over d\theta}\bigg|_{\theta(t-1)}
$$

where $\eta >0$ is our learning rate. We are going in the opposite direction of gradient which helps in getting closer towards the minimum. 

We can also make the derivative of the loss function directly $0$ but it will be hard to calculate the value of $\theta$ directly this way that’s why help of gradient descent is taken.

### Multinomial Logistic Regression

Now $y_i \in \{1,2,\cdots,k\}$. Now we will predict a vector $y_i^* \in \mathbb{R}^k$ for each row such that  

$$
y_i^* = \begin{bmatrix}p(y_i=1\mid x_i),p(y_i=2\mid x_i),\cdots,p(y_i=k\mid x_i)\end{bmatrix}
$$

The one with maximum probability will be the category of that row

With the given training inputs $x_i \in \mathbb{R}^n$ and corresponding observations $y_i \in \mathbb{R}^k$ we need to find $y_i^*$ 

$$
z_i = [z_{i1},z_{i2},\cdots,z_{ik}] = [x_i^T\theta_1,x_i^T\theta_2,\cdots ,x_i^T\theta_k]
$$

where $\theta_j\in \mathbb{R}^{n}$ is the parameter. It can also be redefined as the following 

$$
z_i = x_i^T\Theta
$$

where $\Theta \in \mathbb{R}^{n \times k}$ such that $\Theta = [\theta_1^T,\theta_2^T,\cdots,\theta_k^T]^T$ 

Defining softmax function like sigmoid such that 

$$
y_i^*=\mathrm{softmax}(z_i) =\begin{bmatrix} {\displaystyle \exp{z_{i1}} \over \displaystyle \sum_{j=1}^{k} \exp{z_{ij}}} & {\displaystyle \exp{z_{i2}} \over \displaystyle \sum_{j=1}^{k} \exp{z_{ij}}} & \cdots & {\displaystyle \exp{z_{ik}} \over \displaystyle \sum_{j=1}^{k} \exp{z_{ij}}} \end{bmatrix}
$$

The maximum value of the array $y_i^*$ will be our class.

Here the probability distribution will be 

$$
p(y_i \mid x_i) = \prod_{j=1}^k\hat{y_{ij}}^{y_{ij}}
$$

where $\hat{y_{ij}} = p(y_i = j \mid x_i) = z_{ij}$ and $y_{i1},y_{i2}\cdots y_{ik}$ are such that only one of them will be $1$ and rest of them will be $0$ i.e. $p(y_i=j \mid x_i)$ will have $y_{il} = 0 \forall l\ne j$ and $y_{ij} = 1$.

Defining the loss function 

$$
\begin{align}\mathcal{L}(\Theta) &= -\log p(y_i\mid x_i) \newline &= -\sum_{j=1}^k y_{ij}\log\hat{y}_{ij} \newline &= -\sum_{j=1}^ky_{ij}\big[\log(e^{z_{ij}})-\log\big(\sum_{j=1}^k e^{z_{ij}}\big)\big] \newline &= -\sum_{j=1}^ky_{ij}(z_{ij}-\log\big(\sum_{j=1}^k e^{z_{ij}}\big)) \newline &= -\sum_{j=1}^ky_{ij}(x_i\theta_j - \log\big(\sum_{j=1}^k e^{z_{ij}}\big)) \newline &=-\sum_{j=1}^k y_{ij}(x_i^T\theta_j) + \log\big(\sum_{j=1}^k e^{z_{ij}}\big)\sum_{j=1}^ky_{ij} \newline &= -\sum_{j=1}^k y_{ij}(x_i^T\theta_j) + \log\big(\sum_{j=1}^k e^{z_{ij}}\big) \end{align}
$$

 Finding the gradient of the loss function 

$$
\nabla \mathcal{L}(\Theta) = \begin{bmatrix}{\partial \mathcal{L} \over \partial\theta_1}\newline {\partial \mathcal{L} \over \partial\theta_2} \newline \vdots \newline {\partial \mathcal{L} \over \partial\theta_k}\end{bmatrix}
$$

$$
\begin{align}{\partial \mathcal{L} \over \partial \theta_j} &= -y_{ij}x_i^T + {e^{x_i^T\theta_j}x_i^T \over \sum_{j=1}^ke^{x_i^T\theta_j}}\newline &= (y^*_{ij} - y_{ij})x_i^T \end{align}
$$

where $y_{ij}^* = \displaystyle {e^{x_i^T\theta_j}x_i^T \over \displaystyle \sum_{j=1}^{k} e^{x_i^T\theta_j}}$ which is a similar result as in binomial logistic regression.

Now we can again apply gradient descent and find the best fitting parameters.
