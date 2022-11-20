# Linear Regression

Linear Regression is a type of supervised learning in which we are given some features using them we have to find the respective approximate target values. Example, finding the approximate price of a house (target value) given the size, number of bedrooms etc (features).

Let our training set be $\mathcal{D} = \{(x_1,y_1),(x_2,y_2) \cdots (x_m,y_m)\}$ where $x_m \in \mathbb{R}^n$ are the features and $y_m \in \mathbb{R}$ are our target values. 

As there is a possibility that different input features have extremely different ranges of values, hence the columns are generally normalised i.e. standardising the input values by centring them to result in a zero mean and a standard deviation of one.

With the given training inputs $x_i \in \mathbb{R}^n$ and corresponding observations $y_i \in \mathbb{R}$ we need to find the function $f(x_i)$ such that 

$$
y_i \approx f(x) 
$$

In linear regression one of the basic function can be $f(x_i) = x_i^T\theta$ where $\theta \in \mathbb{R}^n$ is our parameter. So 

$$
y^*_i = x^T_i\theta
$$

where $y^*_i$ is the values predicted by our model.

In predicting values we can add another parameter, the bias i.e.  $y_i^*=x_i^T\theta + b$ where $b$ is our bias. 
Instead of above we can also add a column of ones to the data set. So one of the parameter in $\theta$ will act as the replacement for $b$ . The only difference in below calculations will be the dimension will change from $n \to n+1$

Defining a loss function 

$$
\begin{align}\mathcal{L}(\theta) &= \sum_{i=1}^m(y_i- y^*_i)^2 \newline &=\sum_{i=1}^m(y_i- x_i^T\theta)^2 \newline &=\|Y - X\theta\|^2\end{align}
$$

where $Y = [y_1,y_2,\cdots ,y_m]^T$ i.e. $Y \in \mathbb{R}^{m \times 1}$ and $X = [x_1^Tx_2^T \cdots ,x_m^T]^T$ i.e. $X \in \mathbb{R}^{m \times n}$

So we need to minimise $\mathcal{L}(\theta)$

$$
\begin{align}{d\mathcal{L} \over d\theta} &= {d \over d\theta}\big(\|Y-X\theta\|^2\big)\newline &={d \over d\theta}\big((Y-X\theta)^T(Y-X\theta)\big)\newline &= {d \over d\theta}\big(Y^TY-Y^TX\theta - \theta^TX^TY + \theta^TX^TX\theta\big)\newline &= 0-Y^TX-Y^TX+2\theta^TX^TX\newline &= 2(\theta^TX^TX-Y^TX)\end{align}
$$

We can directly make the derivative $0$ 

$$
\begin{align}2(\theta&^TX^TX-Y^TX) = 0\newline \Rightarrow&\theta^TX^TX = Y^TX\newline
\Rightarrow &X^TX\theta = X^TY\newline
\Rightarrow &\theta = (X^TX)^{-1}X^TY\end{align}
$$

We can only take the inverse of $X^TX$ if $\mathrm{rk}(X^TX) = \mathrm{rk}(X)=n$ where $\mathrm{rk}()$ stands for the rank of the matrix.

Now instead of defining the the function $f(x)$ as $x^T\theta$ we can generalise it by taking  

$$
y^*=f(x) = \phi(x)^T \theta 
$$

where $\phi : \mathbb{R}^n\to\mathbb{R}^k$ could be any non-linear transformation i.e. 

$$
\phi (x) = [\phi_0(x),\phi_1(x) \cdots \phi_{k-1}(x)]^T
$$

with the inputs $x$ because linear regression refers to linearity in parameters. Moreover now $\theta \in \mathbb{R}^k$

Example for performing polynomial regression

$$
\phi(x) = \begin{bmatrix}1 \newline x \newline x^2 \newline \vdots\newline x^{k-1}\end{bmatrix}
$$

Again defining the same loss function as above 

$$
\begin{align}\mathcal{L}(\theta) &= \sum_{i=1}^m(y_i- y^*_i)^2\newline &=\sum_{i=1}^m(y_i- \phi(x_i)^T\theta)^2 \newline &=\|Y - \Phi\theta\|^2\end{align}
$$

where $Y = [y_1,y_2,\cdots ,y_m]^T$ and $\Phi = [\phi(x_1)^T,\phi(x_2)^T, \cdots ,\phi(x_m)^T]^T$ i.e.

$$
\Phi = \begin{bmatrix}\phi_0(x_1) & \cdots &\phi_{k-1}(x_1)\newline \vdots & \ddots & \vdots \newline \phi_0(x_m) &\cdots &\phi_{k-1}(x_m)  \end{bmatrix}
$$

As both $X$ and $\Phi$ are independent of $\theta$ hence 

$$
\theta = (\Phi^T\Phi)^{-1}\Phi^TY
$$

where $\mathrm{rk}(\Phi^T\Phi) = \mathrm{rk}(\Phi) = k$
