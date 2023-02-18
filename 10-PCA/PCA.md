# Principal Component Analysis

Principal Component Analysis is a dimensionality reduction method that is often used to reduce the dimensions of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set in order to have a better visualisation of the data.

Let the data set be $\mathcal{X} = [x_1,\cdots,x_m],x_i\in \mathbb{R}^n$ which has mean $0$. Considering that the data has mean $0$ won’t affect PCA and will make the calculations easier. The covariance matrix of this data will be $(\mu = 0)$

$$
S = {1\over n}\sum_{i=1}^nx_ix_i^T
$$

we assume there exists a low dimensional compressed representation 

$$
z_i = B^Tx_i \in \mathbb{R}^M
$$

where we define the projection matrix(orthogonal matrix), the columns of $B$ would be orthonormal. 

$$
B = [b_1,\cdots,b_m] \in \mathbb{R}^{n\times m}
$$

We can find the variance in the data $z$

$$
\mathbb{V}_z[z] = \mathbb{V}_x[B^T(x-\mu)] = \mathbb{V}_x[B^Tx - B^T\mu] =   \mathbb{V}_x[B^Tx]
$$

As $B$ and $\mu$ are constant hence the $\mathbb{V}_z$ don’t get effect by the mean of the original data, hence we considered it to be $0$. As we considered the mean of the data $0$, it turns out that mean of the projected data is also $0$

$$
\mathbb{E}_z[z]= \mathbb{E}_x[B^Tx] = B^T\mathbb{E}_x[x] = 0
$$

We maximise the variance in order to preserve as much variability (or information) as possible during the process of reducing the dimension of the data.   

We start by seeking a single vector $b_1 \in \mathbb{R}%D$ that maximises the variance of the projected data, i.e. we aim to maximise the variance of the first co-ordinate $z_1$ of $z\in \mathbb{R}^M$

$$
V_1 = \mathbb{V}[z_1] = {1\over m}\sum_{i=1}^mz_{1i}^2\\
z_{1i} = b_1^Tx_n
$$

$$
V_1 = {1\over m}\sum_{i=1}^m (b_1^Tx_i)^2 = {1\over m}\sum_{i=1}^mb_1^Tx_ix_i^Tb_1\\
= b_1^T\bigg({1\over m} \sum_{i=1}^Nx_ix_i^T\bigg)b_1 = b_1^TSb_1
$$

So our constrained optimisation problem is 

$$
\begin{gathered}
\max_{b_1} b_1^TSb_1\newline
\ce{subject to} \quad \|b_1\|^2 =1
\end{gathered}
$$

so we can use the Lagrangian multiplier and maximise it 

$$
\begin{gathered}
\mathcal{L}(b_1,\lambda_1) = b_1^TSb_1+\lambda_1(1-b_1^Tb_1)
\end{gathered}
$$

$$
\begin{gathered}
{\partial \mathcal{L} \over \partial b_1} = 2b_1^TS-2\lambda_1b_1^T = 0\\
Sb_1 = \lambda_1b_1 
\end{gathered}
$$

$$
\begin{gathered}
{\partial \mathcal{L} \over \partial \lambda_1} = 1-b_1^Tb_1\\
b_1^Tb_1=1
\end{gathered}
$$

Hence, $b_1$ is an eigenvector of the data covariance matrix $S$ and the Lagrange multiplier $\lambda_1$ plays the role of the corresponding eigenvalue which would be equal to the variance 

$$
V_1 = b_1^TSb_1 = \lambda_1b_1^Tb_1 = \lambda_1
$$

So for creating the matrix $B$ in order to maximise the variance in that dimension, we need to take the $m$ eigenvectors associated with the largest eigenvalues of the covariance matrix $S$

Thus the steps for performing PCA are

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/10-PCA/images/Untitled.png" width="200"/>
</p>

Original Data

1. Centre the original data by subtracting  the mean from the data set

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/10-PCA/images/Untitled 1.png" width="200"/>
</p>

1. Divide by the standard deviation so that variance become $1$ along each axis
2. Compute the covariance matrix of the data
3. Compute the eigenvalues and eigenvectors of the covariance matrix in order to compute data

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/10-PCA/images/Untitled 2.png" width="200"/>
</p>

1. Project the data onto the space whose basis are the first $M$ eigenvectors

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/10-PCA/images/Untitled 3.png" width="200"/>
</p>

1. Undo the standardisation by again multiplying the standard deviation and adding the mean

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/10-PCA/images/Untitled 4.png" width="200"/>
</p>

### Questions

1. **What is the Curse of Dimensionality ?**
    
    The more the number of features, more the model starts becoming complex which results in high chances of overfitting because the model trained on it will be a lot dependent on the training data. Hence it won’t perform good on the test data set.
    
2. ****Why do we standardisation in PCA ?****
    
    Standardisation is done because if we don’t do it, features with larger ranges of numbers will have higher co-variances.
    
3. **What are the limitations of PCA ?**
    
    PCA assumes a linear relationship between features hence doesn’t work for non-linear data. Moreover, it assumes a correlation between features i.e. if the features are not correlated, PCA will be unable to determine principal components. if all eigenvalues are roughly equal then we can not select the principal components.
