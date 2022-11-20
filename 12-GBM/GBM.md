# Gradient Boosting Machine

Gradient Boosting is a type of supervised learning algorithm. Unlike gradient descent in which we explicitly compute the gradient of the loss function to find the minimum, in gradient boosting we learn an approximate gradient by training a weak model to fit the data or more precisely residual data.

### Regression with GBM

Let our data $\mathcal{D} = \{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$ where $x_i \in \R^n$ and $y_i \in \R$

As we are performing regression so we can define our loss function as least squared error i.e. 

$$
\mathcal{L}(y_i,f(x_i)) = {1\over 2}[y_i-f(x_i)]^2
$$

We initialise our model with a constant value such that 

$$
f_0(x) = \argmin_\gamma\sum_{i=1}^m\mathcal{L}(y_i,\gamma) 
$$

$$
L(y,\gamma) = {1\over 2}\sum_{i=1}^m[y_i-\gamma]^2\\
{\partial L \over \partial \gamma} = -\sum_{i=1}^m(y_i-\gamma)=0\\
\gamma = {1\over m}\sum_{i=1}^my_i
$$

Hence, $f_0(x)$ is defined as the average of the target values. 

Now we compute the pseudo residual vector $r \in \R^n$ i.e. 

$$
\begin{align*}r_i &= -{\partial \mathcal{L(y_i,F(x_i))} \over \partial f(x_i)}\bigg|_{\mathcal{F}(x_i) =f_0(x_i)}\\
&\Rightarrow r_i = y_i -\mathcal{F}(x_i) \end{align*}
$$

Now we build our weak model, e.g. a decision tree with less max depth. We build our decision tree on the residual data i.e. $\mathcal{D_1} = \{(x_1,r_1),(x_2,r_2),\cdots(x_m,r_m)\}$  i.e. the actual target vector are replaced by the pseudo residual vector $r$. 

As there would be more than one data point present in some of the leaf nodes hence we have to find a best fitting value $\gamma_j$ for each leaf node.

$$
\gamma_j = \argmin_\gamma \sum_{x_i \in l_j}\mathcal{L}(y_i,f_0(x_i)+\gamma)
$$

As this is a similar loss function as above hence this will also be the average i.e. 

$$
\gamma_j = {1\over \#l_i}\sum_{x_i\in l_i}y_i -f_0(x_i) 
$$

where $\#l_i$ is the number of data points present in the leaf node 

Now we update our function i.e. 

$$
f_1(x) = f_0(x) + \eta \gamma
$$

where the value of $\gamma$ is the value associated to that leaf node in which $x_i$ is present

Now the above process is repeated the only difference is $f_0(x)$ is changed by $f_1(x)$. Then this is continued $M$ times i.e. a total of $M$ decision trees are created and we get our final prediction function $f_M(x)$

### Binary Classification with GBM

We will have the same properties of data set as above the only difference will be now $y_i \in \{0,1\}$

As we are performing binary classification so we can define our loss function as cross entropy i.e. 

$$
\mathcal{L}(y_i,f(x_i)) = -(y_i\log \hat{y_i} +(1-y_i)\log(1-\hat{y_i}))
$$

where $\hat{y_i} = p(y_i=1\mid x_i)= {1\over 1 + e^{-\ln (\ce{odds})}}$ where $\ce{odds} = {p(y_i = 1 \mid x_i) \over p(y_i = 0 \mid x_i)}$ and $f(x_i)= \ln(\ce{odds})$ so the above loss function can also be written as 

$$
\mathcal{L}(y_i,f(x_i)) = -y_if(x_i)+\log(1+e^{f(x_i)})
$$

As above we follow similar steps i.e. we initialise our model with a constant value such that 

$$
f_0(x) = \argmin_\gamma\sum_{i=1}^m\mathcal{L}(y_i,\gamma) 
$$

$$
L(y,\gamma) =- \sum_{i=1}^my_i\gamma+\ln(1+e^{\gamma}) \\{\partial L \over \partial \gamma}= -\sum_{i=1}^my_i+{e^\gamma\over 1+e^\gamma}=0\\
\Rightarrow \gamma = \ln({p\over 1-p})\\
p = {1\over m}\sum_{i=1}^my_i\\

$$

Hence, $f_0(x)$ is defined as the average of the target values. 

Computing the pseudo residual vector $r \in \R^n$ i.e. 

$$
\begin{align*}&r_i = -{\partial \mathcal{L(y_i,F(x_i))} \over \partial f(x_i)}\bigg|_{\mathcal{F}(x_i) =f_0(x_i)}\\
\Rightarrow &r_i = y_i -{1\over 1+e^{-\mathcal{F}(x_i)}}\bigg| _{\mathcal{F}(x_i) =f_0(x_i)}\\ \Rightarrow & r_i =y_i-\hat{y}_i\end{align*}
$$

We build our decision tree on the residual data i.e. $\mathcal{D_1} = \{(x_1,r_1),(x_2,r_2),\cdots(x_m,r_m)\}$  i.e. the actual target vector are replaced by the pseudo residual vector $r$. 

As there would be more than one data point present in some of the leaf nodes hence we have to find a best fitting value $\gamma_j$ for each leaf node.

$$
\gamma_j = \argmin_\gamma \sum_{x_i \in l_j}\mathcal{L}(y_i,f_0(x_i)+\gamma)
$$

But here finding the derivative and making it zero is tough, hence its better to apply gradient descent where   

$$
{\partial \mathcal{L} \over \partial \lambda} = -y_if(x_i) + {\exp({f(x_i) + \lambda)}\over {\exp{(f(x_i) + \lambda)} +1 }}
$$

After some iteration of gradient descent we can get a decent value of $\lambda_j$ 

Now we update our function i.e. 

$$
f_1(x) = f_0(x) + \eta \gamma
$$

where the value of $\gamma$ is the value associated to that leaf node in which $x_i$ is present

Thus similarly as above we will repeat this above process with the only difference that instead of $f_0(x)$ not there would be $f_1(x)$ and after $M$ iterations we will get the function $f_M(x)$ and we can find the probability $\hat{y}$ as  

$$
\hat{y} = {1 \over 1 + e^{-f_M(x)}}
$$

and we can make our predictions as 

$$
y^* = \begin{cases} 1 & \hat{y} > 0.5\\ 0 & \ce{otherwise}\end{cases}
$$

Below is an embedded link of a website which have an interactive playground for Gradient Boosting

[http://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html](http://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html)

### Questions

1. **How are weak decision trees created ?**
    
    By making the maximum depth of the decision tree small.
    
2. **How is GBM different from Gradient descent ?**
    
    Gradient descent is used to find the parameters associated with a single model that optimises some loss function. Whereas, GBM  consists of multiple weak models whose output is added together to get an overall prediction. The gradient descent occurs on the output of the model and not the parameters of the weak models.