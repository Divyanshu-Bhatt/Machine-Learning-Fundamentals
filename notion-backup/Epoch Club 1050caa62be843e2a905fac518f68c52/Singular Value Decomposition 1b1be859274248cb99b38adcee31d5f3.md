# Singular Value Decomposition

Let any matrix $A \in \R^{m \times n}$. Let $v_1 \in \R^n$ be an unit vector in the row space of $A$. Then we can define a vector $u_1\in \R^m$ in the column space of $A$ such that  

$$
\sigma_1u_1 = Av_1
$$

where $\sigma_1$ is such that $u_1$ is an unit vector

<aside>
📌 Row space of a matrix is the space spanned by the row vectors of matrix. Similarly, Column space of a matrix is the space spanned by the column vectors of matrix.

</aside>

Let $r$  be the rank of matrix $A$ then $r = \dim C(A) = \dim R(A)$  where $C(A)$ and $R(A)$ are the column and the row space of the matrix respectively. 

Let $(v_1,v_2\cdots ,v_r)$ be the orthonormal basis of $R(A)$ then  

$$
\sigma_i u_i = Av_i \quad \forall i\in \{1,2,\cdots, r\}
$$

$(u_1,u_2,\cdots,u_r)$ is the orthonormal basis of $C(A)$ because linear transformation doesn’t change angles between two vectors.

$\dim N(A) = n - \dim R(A) = n-r$ where $N(A)$ is the null space of $A$. So we can define the vectors $(v_{r+1},\cdots,v_n)$ as the orthonormal basis of $N(A)$.

As $R(A)$ is perpendicular to $N(A)$ hence  

$$
V = [v_1,v_2,\cdots ,v_n]
$$

the matrix $V$ results in an orthogonal matrix. 

Similarly, $\dim N(A^T) = m - \dim C(A) = m-r$ where $N(A^T)$ is the null space of $A^T$ or the left null space of $A$. So we can define the vectors $(u_{r+1},\cdots,u_m)$ as the orthonormal basis of $N(A^T)$. 

As $C(A)$ is perpendicular to $N(A^T)$ hence  

$$
U = [u_1,u_2,\cdots,u_m]
$$

the matrix $U$ results in an orthogonal matrix 

With our starting condition that $\sigma u_i = Av_i$  we can write

$$
AV = U\Sigma
$$

where $\Sigma \in \R^{m \times n}$ is a kind of diagonal matrix such that all the diagonal elements are equal to $\sigma_i$ $\forall i = \{1,\cdots,r\}$ and rest of the diagonal and non diagonal elements are zero. 

$\ce{Example:}$

Following is a $\Sigma$ matrix when $m > n$ and $\mathrm{rk}(A) = n$ 

$$
\Sigma = \left[\begin{array}{cc}\sigma_1 & \cdots & 0 \\\vdots & \ddots & \vdots \\ 0 & \cdots & \sigma_n\\
\vdots & \ddots & \vdots \\
0 & \cdots & 0\end{array}\right]
$$

Following is a $\Sigma$ matrix when $n > m$ and $\mathrm{rk}(A) = m$ 

$$
\Sigma = \left[\begin{array}{cc}\sigma_1 & \cdots & 0 & \cdots  & 0\\\vdots & \ddots & \vdots & \ddots &\vdots \\ 0 & \cdots & \sigma_m
&\cdots & 0\end{array}\right]
$$

So we have found the relation 

$$
A = U \Sigma V^{-1}\\
A = U\Sigma V^T
$$

The above decomposition is known as the singular value decomposition

<aside>
📌 $V$ is an orthonormal matrix hence $V^T = V^{-1}$

</aside>

One of the most important thing kept in mind while constructing the matrices $U$ and $V$ is that. The vectors $u_1$ and $v_1$ and so on are taken such that $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r$ 

Following is an illustration of how the transformation takes place through SVD

![$A \in \R^{3\times2}$   $V^T$performs a basis change in $\R^2$
$\Sigma$ scales and maps from $\R^2$ to $\R^3$
The ellipse in bottom right lives in $\R^3$
$U$ performs a basis change within $\R^3$](Singular%20Value%20Decomposition%201b1be859274248cb99b38adcee31d5f3/image.png)

$A \in \R^{3\times2}$   $V^T$performs a basis change in $\R^2$
$\Sigma$ scales and maps from $\R^2$ to $\R^3$
The ellipse in bottom right lives in $\R^3$
$U$ performs a basis change within $\R^3$

Hence we can write the matrix $A$ as $U\Sigma V^T$ or 

$$
A = \sum_{i=1}^r u_i\sigma_iv_i^T
$$

the above formula can be directly computed from $U\Sigma V^T$. As the null space vectors are not important for us because they don’t contribute anything to the above summation we can also write 

$$
A = \tilde{U}\tilde{\Sigma}\tilde{V}^T
$$

where $\tilde{U} =  [u_1,\cdots,u_r] \in \R^{n\times r}$, $\tilde{\Sigma} = \ce{diag([\sigma_1,\cdots ,\sigma_r])}\in \R^{r\times r}$ and $\tilde{V} = [v_1,\cdots,v_r] \in \R^{m\times r}$. This is called as economy SVD

For calculating SVD, we construct a symmetric, positive semi-definite matrix $A^TA \in \R^{n\times n}$. As it is a square matrix hence we can do its eigen decomposition also

$$
A^TA = PDP^T\\
(U\Sigma V^T)^T(U\Sigma V^T) = PDP^T\\
PDP^T = V \left[\begin{array}{cc} \sigma_1^2  & \cdots & \cdots&\cdots&0\\ \vdots  & \ddots &&&\vdots \\ \vdots  &  & \sigma_r^2&&\vdots\\ \vdots &&&\ddots& \vdots\\ 0 & \cdots&\cdots&\cdots&0\end{array}\right]V^T
$$

hence $\sigma_i$ are the squares of the eigenvalues of $A^TA$ and $V$ are the eigenvectors of the matrix $A^TA$. Now with the $v_i$’s we can find $u_i$’s as following 

$$
u_i = {Av_i \over \| Av_i\|} = {1 \over \sqrt{\lambda_i}}Av_i = {1 \over \sigma_i}Av_i 
$$

<aside>
📌 $v_1,\cdots,v_n$ are the eigenvectors of $A^TA$ and
$u_1,\cdots,u_n$ are the eigenvectors of $AA^T$ and
$\sigma_1^2,\cdots,\sigma_r^2$ are the eigenvalues of the above two matrices

</aside>

### Questions

1. **What  are the matrices $U,\Sigma$ and $V$ responsible for in the linear transformation ?**
    
    The $V^T$ matrix changes the basis from one to another keeping the co-ordinates same. $\Sigma$ changes the dimensional space from one to other with scaling. Then $U$ matrix again changes the basis keeping the co-ordinate same like $U$  
    
2. **What is Matrix Approximation ? How can it be helpful ?**
    
    Any matrix of rank $r$, can be written as the sum of $r$ rank one matrices i.e. 
    
    $$
    A = \sum_{i = 1}^r \sigma_iu_iv_i^T
    $$
    
    Let $k$ be such that $k<r$ then we can write
    
    $$
    \tilde{A}(k) = \sum_{i = 1}^k \sigma_iu_iv_i^T
    $$
    
    So $\tilde{A}(k)$ is the best approximation of $A$ which has a rank $k$
    
    It helps in compressing data like some of the high-dimensional images can be blurred or can be compressed by just storing the first few matrices and removing the rest and there would not be much difference in the quality.