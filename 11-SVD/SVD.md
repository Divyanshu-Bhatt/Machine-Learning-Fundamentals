# Singular Value Decomposition

Let any matrix $A \in \mathbb{R}^{m \times n}$. Let $v_1 \in \mathbb{R}^n$ be an unit vector in the row space of $A$. Then we can define a vector $u_1\in \mathbb{R}^m$ in the column space of $A$ such that  

$$
\sigma_1u_1 = Av_1
$$

where $\sigma_1$ is such that $u_1$ is an unit vector

Row space of a matrix is the space spanned by the row vectors of matrix. Similarly, Column space of a matrix is the space spanned by the column vectors of matrix.

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

where $\Sigma \in \mathbb{R}^{m \times n}$ is a kind of diagonal matrix such that all the diagonal elements are equal to $\sigma_i$ $\forall i = \{1,\cdots,r\}$ and rest of the diagonal and non diagonal elements are zero. 

$\ce{Example:}$

Following is a $\Sigma$ matrix when $m > n$ and $\mathrm{rk}(A) = n$ 

$$
\Sigma = \left[\begin{array}{cc}\sigma_1 & \cdots & 0 \newline \vdots & \ddots & \vdots \newline 0 & \cdots & \sigma_n\newline
\vdots & \ddots & \vdots \newline
0 & \cdots & 0\end{array}\right]
$$

Following is a $\Sigma$ matrix when $n > m$ and $\mathrm{rk}(A) = m$ 

$$
\Sigma = \left[\begin{array}{cc}\sigma_1 & \cdots & 0 & \cdots  & 0\newline \vdots & \ddots & \vdots & \ddots &\vdots \newline 0 & \cdots & \sigma_m
&\cdots & 0\end{array}\right]
$$

So we have found the relation 

$$
A = U \Sigma V^{-1}\\
A = U\Sigma V^T
$$

The above decomposition is known as the singular value decomposition

$V$ is an orthonormal matrix hence $V^T = V^{-1}$

One of the most important thing kept in mind while constructing the matrices $U$ and $V$ is that. The vectors $u_1$ and $v_1$ and so on are taken such that $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r$ 

Following is an illustration of how the transformation takes place through SVD

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/11-SVD/images/visualise.png" width="300"/>
</p>

$A \in \mathbb{R}^{3\times2}$   $V^T$performs a basis change in $\mathbb{R}^2$
$\Sigma$ scales and maps from $\mathbb{R}^2$ to $\mathbb{R}^3$
The ellipse in bottom right lives in $\mathbb{R}^3$
$U$ performs a basis change within $\mathbb{R}^3$

Hence we can write the matrix $A$ as $U\Sigma V^T$ or 

$$
A = \sum_{i=1}^r u_i\sigma_iv_i^T
$$

the above formula can be directly computed from $U\Sigma V^T$. As the null space vectors are not important for us because they don’t contribute anything to the above summation we can also write 

$$
A = \tilde{U}\tilde{\Sigma}\tilde{V}^T
$$

where $\tilde{U} =  [u_1,\cdots,u_r] \in \mathbb{R}^{n\times r}$, $\tilde{\Sigma} = \ce{diag([\sigma_1,\cdots ,\sigma_r])}\in \mathbb{R}^{r\times r}$ and $\tilde{V} = [v_1,\cdots,v_r] \in \mathbb{R}^{m\times r}$. This is called as economy SVD

For calculating SVD, we construct a symmetric, positive semi-definite matrix $A^TA \in \mathbb{R}^{n\times n}$. As it is a square matrix hence we can do its eigen decomposition also

$$
\begin{gather}
A^TA = PDP^T\newline
(U\Sigma V^T)^T(U\Sigma V^T) = PDP^T\newline
PDP^T = V \left[\begin{array}{cc} \sigma_1^2  & \cdots & \cdots&\cdots&0\newline \vdots  & \ddots &&&\vdots \newline \vdots  &  & \sigma_r^2&&\vdots\newline \vdots &&&\ddots& \vdots\newline 0 & \cdots&\cdots&\cdots&0\end{array}\right]V^T
\end{gather}
$$

hence $\sigma_i$ are the squares of the eigenvalues of $A^TA$ and $V$ are the eigenvectors of the matrix $A^TA$. Now with the $v_i$’s we can find $u_i$’s as following 

$$
u_i = {Av_i \over \| Av_i\|} = {1 \over \sqrt{\lambda_i}}Av_i = {1 \over \sigma_i}Av_i 
$$

$v_1,\cdots,v_n$ are the eigenvectors of $A^TA$ and
$u_1,\cdots,u_n$ are the eigenvectors of $AA^T$ and
$\sigma_1^2,\cdots,\sigma_r^2$ are the eigenvalues of the above two matrices
