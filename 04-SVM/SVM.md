# Support Vector Machine

Support vector machine is a type of supervised learning. It helps in doing binary classification of data. It maps the input feature to a higher dimension and then find a linear hyperplane that separates the two classes which would result in a non-linear decision boundary in the lower dimension.

Let $\mathcal{D} = \{(x_1,y_1)\cdots,(x_m,y_m)\}$ where $x_i \in \mathbb{R}^n$ and $y_i \in \{-1,1\}$ where $-1,1$ represents the two different classes. Lets consider that for now the data is linearly separable

So we try to find such hyperplane which separates the two data. 

$$
f(x) =\langle w,x\rangle + b
$$

where $w \in \mathbb{R}^n$ and $b\in \mathbb{R}$. The hyperplane has the normal vector $w$ and the intercept $b$

Moreover $w$ and $b$ satisfies the following condition

$$
y_n(\langle w,x_n\rangle +b)\ge 0 \Leftrightarrow \begin{cases}\langle w,x_n\rangle + b \ge 0 & \text{when} \ y_n=1\newline \langle w,x_n\rangle + b \le 0 & \text{when} \ y_n=-1\end{cases}
$$

But there would be infinite such $w$ present, so one of the best idea is to choose the separating hyperplane that maximises the margin between the two classes i.e. the positive and negative classes.

### Maximum Margin Classifier

Margin is the distance of the separating hyperplane to the closest examples in the data set, assuming that the data set is linearly separable. 

Lets consider an example $x_a$ (the closest example). Without the loss of generality we can consider the example $x_a$ to be on the positive side of the hyperplane. So we want to calculate its distance from the hyperplane

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/04-SVM/images/margin1.png" width="300"/>
</p>

$$
x_a = x_a' + r{w \over \|w\|}
$$

So the distance $r$ is the margin. We want the positive examples to be further than $r$ from the hyperplane and the negative examples to be further than distance $r$ in negative direction hence 

$$
y_n(\langle w,x_n\rangle + b) \ge r
$$

As for the hyperplane we only need the direction of $w$ and not the magnitude. Hence, we can make another condition for making the calculations easier i.e. $\|w\|  =1$, so that $r$ is just the scaling factor. Thus our optimising problem becomes

$$
\max_{w,b,r}{ r}\newline
\ce{subject to} \ {y_n(\langle w,x_n \rangle + b) \ge r},\quad {\|w\| = 1}, \quad r>0
$$

Another method is to make the margin $1$ rather normalising the vector 

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/04-SVM/images/margin2.png" width="300"/>
</p>

From this figure, as $x'_a$ lies on the hyperplane hence it would satisfy its equation i.e. 

$$
\langle w,x'_a\rangle + b= 0\newline
\left\langle w,x_a-r{w\over \|w\|}\right\rangle + b = 0\newline \langle w,x_a\rangle + b + \left\langle w,-r{w\over \|w\|}\right\rangle = 0\\ \left\langle w,r{w\over \|w\|}\right\rangle= 1\newline
r ={1\over \|w\|}
$$

So the optimisation problem changes as 

$$
\max_{w,b}{ {1 \over \|w\|}}\\
\ce{subject to} \ {y_n(\lang w,x_n \rang + b) \ge 1} , \quad r>0
$$

The above would be same as 

$$
\min_{w,b}{ {1\over2}\|w\|^2}\\
\ce{subject to} \ {y_n(\lang w,x_n \rang + b) \ge 1}
$$

Using Lagrange multiplier we can define a single loss function as follows 

$$
\mathcal{L}(w,b\,\alpha) = {1\over 2}\|w\|^2-\sum_{i=1}^n \alpha_i [y_i(\lang w,x_i\rang+b)-1]
$$

where $\alpha_i \ge 0 \forall i =\{1,2,\cdots n\}$ i.e. $\alpha = [\alpha_1,\cdots ,\alpha_n]$

Finding the partial derivative of the loss function and making them $0$

$$
\begin{align*}{\partial \mathcal{L}\over \partial w} &= w^T - \sum_{i=1}^n \alpha_iy_ix^T_i=0\\&\Rightarrow w = \sum_{i=1}^n\alpha_iy_ix_i
\end{align*}
$$

$$
\begin{align*}{\partial \mathcal{L}\over \partial b} &=-  \sum_{i=1}^n \alpha_iy_i=0\end{align*}
$$

Substituting the above values in the loss function 

$$
\begin{align*}\mathcal{L}(w,b,\alpha) &= {1\over2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\lang x_i,x_j\rang - \sum_{i=1}^n\alpha_i\bigg[y_ix^T_i\bigg(\sum_{j=1}^n \alpha_jy_jx_j\bigg)-y_ib-1\bigg]
\\&=\sum_{i}^n\alpha_i -{1\over2}\sum_{i,j=1}^n \alpha_i\alpha_jy_iy_j\lang x_i,x_j\rang - b\sum_{i=1}^n\alpha_iy_i\end{align*}
$$

So the optimisation problem changed as (there is additional negative sign below because we maximise the Lagrange function, so for making it a minimum question, added a negative sign)

$$
\min   {1\over2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\lang x_i,x_j\rang - \sum_{i=1}^n\alpha_i\\
\ce{subject to} \quad \sum_{i=1}^n\alpha_iy_i =0,\\
\alpha \ge0
$$

### Soft Margin Classifier

The above method we applied is also called as hard margin classifier. The main problem with that is the decision boundary will change a lot because of the outliers. For example below the green line seems to be a better decision boundary than the red line for generalising but according to the above example red line would be selected because it will have the minimum error. 

![SVM.drawio (2).svg](Support%20Vector%20Machine%20a9aa83a697564c03a5a40075a614da74/SVM.drawio_(2).svg)

So we use Soft margin classifier in order to prevent this. Soft margin classifier allows some miss classifications. So the loss function we define is not only based on the miss classification but also how much away it is from the considered decision boundary 

![image.png](Support%20Vector%20Machine%20a9aa83a697564c03a5a40075a614da74/image%202.png)

So our optimisation problem becomes 

$$
\min_{w,b} {1\over 2}\|w\|^2 + C\sum_{i=1}^n\xi_i\\
\ce{subject to} \  {y_n(\lang w,x_n \rang + b) \ge 1 - \xi},\quad \xi \ge 0
$$

where $C$ is a regulariser parameter which can be found using cross validation and $\xi_i$ is defined as follows 

$$
\xi_i = \max \{0,1-y_i(\lang w,x_i\rang + b)\}
$$

So if the point classified is such that $y_i(\lang w,x_i\rang +b) \ge 1$ then the loss by $\xi$ will be $0$. But if its not then the loss will be $1-y_i(\lang w,x_i\rang +b)$ the distance as shown in the above figure. 

Even if there are points that are classified correctly but if they are too close to the hyperplane selected, they will contribute to the loss function.

Applying the Lagrange multiplier for the above 

$$
\begin{align*}\mathcal{L}(w,b,\xi,\alpha, \gamma) = {1\over 2}&\|w\|^2+C\sum_{i=1}^n \xi_i \\&-\sum_{i=1}^n\alpha_i [y_i(\lang w,x_i\rang+b)-1 +\xi_i]-\sum_{i=1}^n\gamma_i\xi_i\end{align*}
$$

where $\alpha_i,\gamma_i \ge 0 \forall i =\{1,2,\cdots n\}$ and $\alpha = [\alpha_1,\cdots ,\alpha_n]$, $\gamma = [\gamma_1,\cdots ,\gamma_n]$  

Finding the partial derivative of the loss function and making them $0$

$$
\begin{align*}{\partial \mathcal{L}\over \partial w} &= w^T - \sum_{i=1}^n \alpha_iy_ix^T_i=0\\&\Rightarrow w = \sum_{i=1}^n\alpha_iy_ix_i
\end{align*}
$$

$$
\begin{align*}{\partial \mathcal{L}\over \partial b} &=-  \sum_{i=1}^n \alpha_iy_i=0\end{align*}
$$

$$
\begin{align*}{\partial \mathcal{L}\over \partial \xi_i} &=  C-\alpha_i-\gamma_i = 0\\ &\Rightarrow \alpha_i + \gamma_i = C\end{align*}
$$

Substituting the above three conditions into the Lagrange multiplier

$$
\begin{align*}\mathcal{L}(b,\alpha,\gamma) =\sum_{i}^n\alpha_i -{1\over2}\sum_{i,j=1}^n \alpha_i&\alpha_jy_iy_j\lang x_i,x_j\rang - b\sum_{i=1}^n\alpha_iy_i\\ &+\sum_{i=1}^n(C-\alpha_i-\gamma_i)\xi_i \end{align*}
$$

So the optimisation problem changed as 

$$
\min   {1\over2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\lang x_i,x_j\rang - \sum_{i=1}^n\alpha_i\\
\ce{subject to} \quad \sum_{i=1}^n\alpha_iy_i =0,\\
0 \le\alpha_i \le C
$$

As $\alpha_i,\gamma_i \ge 0$ and $\alpha_i + \gamma_i = C$

$$
C-\alpha_i =\gamma_i \ge 0\\
C-\alpha_i \ge 0\\
\alpha_i \le C
$$

Hence the only difference between the final optimisation function in the hard and soft classifier is $\alpha$ gets a upper limit.

Now we can apply gradient descent in the above function and find the minimum 

Now this above optimisation problem can be solved using SMO (Sequential minimal optimisation)

### Kernel Trick

Now if we can’t create a linear hyperplane with the given feature, we try to transform it into a higher dimension so that we can get a linear decision boundary in the transformed space.

So we can define a function $\phi$ such that $x \mapsto \phi(x)$ in the transformed space. So in our optimisation problem instead of finding $\lang x_i,x_j\rang$ we have to find $\lang \phi(x_i) ,\phi(x_j)\rang$. But doing this directly will take a lot of computation time so we apply the kernel trick.

$\ce{Example:}$

Let 

$$
\phi(x) = \begin{bmatrix}x_1x_1\\x_1x_2\\x_1x_3\\x_2x_1\\x_2x_2\\x_2x_3\\x_3x_1\\x_3x_2\\x_3x_3\end{bmatrix}
$$

So instead of first transforming we can also write 

$$
\phi(x_i)^T\phi(x_j) = ( x_i^Tx_j)^2
$$

This would take a lot less computation rather than first transforming and then finding the inner product.

Similarly suppose we want all the possible combination where the polynomial has degree $d$ we can be done as 

$$
\phi(x_i)^T\phi(x_j) = (x_i^Tx_j)^d
$$

If we also wanna add individual terms like $x_1,x_2\cdots x_n$ as one of the dimension of $\phi(x)$ we can write 

$$
\phi(x_i)^T\phi(x_j) = ( x_i^Tx_j +c)^d
$$

$x_1,x_2$ will come but with some coefficient 

### Validity of Kernel Function

The necessary and the sufficient condition for the kernel is that the matrix $K$ should be a semi positive definite matrix where $K \in \R^d$ for $d$ different data points

$$
K = \begin{bmatrix}k(x_1,x_1)&k(x_1,x_2)&\cdots& k(x_1,x_d)\\k(x_2,x_1)& k(x_2,x_2)& \cdots & k(x_2,x_d)\\\vdots & \ddots & & \vdots\\ \vdots & & \ddots & \vdots \\ k(x_d,x_1) & k(x_d,x_2) & \cdots & k(x_d,x_d) \end{bmatrix}
$$

### Questions

1. **What is SMO?**
    
    SMO or Sequential Minimal Optimisation is an algorithm that is used to solve quadratic programming. We try to find the parameters that makes the loss function minimum by fixing all the parameters except one and try to find the optimum value for that parameter and then we do this for the next one. We continue to do this until convergence.
    
2. **Why do we apply the kernel trick ?**
    
    The kernel trick helps us to calculate the inner product of the feature must efficiently in the higher dimension
    
3. **How is Soft margin better than Hard margin SVM ?**
    
    Soft margin SVM is better that Hard margin SVM because even a single outlier can make a massive change in the decision boundary, which makes the classifier overly sensitive towards the noise in the data.
