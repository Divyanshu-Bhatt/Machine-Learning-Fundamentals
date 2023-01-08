# T-Stochastic Neighbour Embedding

T-SNE is a dimensionality reduction method that is often used to reduce the dimensions of large data sets that still contains most of the information of the large set in order to have a better visualisation of the data.

Let our data set $\mathcal{D}$ contain the data points $\{x_1,\cdots,x_m\}$. It starts by converting the high dimensional Euclidean distance between the data points into conditional probabilities that represents similarities. The similarity of data point $x_j$ with respect to $x_i$ is $p_{j\mid i}$ which is calculated using Gaussian Kernel. Moreover, as we are only interested in pairwise similarity hence $p_{i\mid i}$ is consider to be $0$. 

$$
p_{j\mid i} = {\displaystyle\exp\bigg({-\|x_i-x_j\|^2\over 2 \sigma_i^2}\bigg) \over \displaystyle\sum_{k\ne i}\exp\bigg({-\|x_i-x_k\|^2\over 2 \sigma_i^2}\bigg) }
$$

$\sigma_i$ is the variance of the Gaussian Kernel which is an user defined value. It is chosen to achieve the desire perplexity $(\mathcal{P})$ 

$$
\mathcal{P} = 2^{\mathcal{H}}\\
\mathcal{H} = -\sum_{j\ne i}p_{j\mid i}\log_2p_{j\mid i}
$$

Generally $\mathcal{P} = 30$

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/09-Tsne/images/perplexity.png" width="500"/>
</p>

Data Visualisation on the MNIST data set with different perplexity

To make the optimisation easier probabilities we make the probability symmetrical i.e. we calculate $p_{ij}$ as 

$$
p_{ij} = {p_{j\mid i}+p_{i\mid j}\over 2m}
$$

Let $\tilde{\mathcal{D}}=\{y_1,y_2,\cdots,y_m\}$ are the low dimensional representation of the same points

Now we calculate the similarities between the data points in this low dimension. But in the low dimensional map we can use a  probability distribution that has much heavier tails than a Gaussian to convert distance into probabilities. So instead of using Gaussian Kernel we use the T-kernel. 

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/09-Tsne/images/distribution.png" width="300"/>
</p>

Similarly we calculate $q_{ij}$

$$
q_{ij} = {(1 + \|y_i-y_j\|^2)^{-1}\over \displaystyle\sum_{k\ne j}(1+\|y_k-y_l\|^2)^{-1}} = {w_{ij}\over Z}
$$

So we create a loss function as the Kullback-Leibler divergence between pairwise similarities in the high-dimensional and in the low-dimensional spaces. This works as close neighbours in the high dimension attract each other while all the points repulse each other, i.e. 

$$
\begin{align*}L &= \sum_{i,j} p_{ij}\log{p_{ij} \over q_{ij}}\\&= \sum_{ij} p_{ij}\log p_{ij} - \sum_{i,j}p_{ij}\log q_{ij}\\
&= \sum_{ij} p_{ij}\log p_{ij} - \sum_{i,j}p_{ij}\log w_{ij} + \sum_{i,j}p_{ij}\log Z \\
&= \sum_{ij} p_{ij}\log p_{ij} - \sum_{i,j}p_{ij}\log w_{ij} + \log Z\sum_{i,j}p_{ij}\\ 
 &=\sum_{ij} p_{ij}\log p_{ij} - \sum_{i,j}p_{ij}\log w_{ij} + \log Z
\end{align*}
$$

$p_{ij}$ were defined such that $\sum_{i,j} p_{ij} = 1$ 

As the first is constant hence we can redefine the loss function as  

$$
\mathcal{L} = -\sum_{ij}p_{ij}\log w_{ij} + \log \sum_{ij}w_{ij}
$$

Now we can apply gradient descent on the above loss function where ${\partial \mathcal{L} \over \partial y_i}$ is a following

$$
\begin{gather}{\partial\mathcal{L} \over \partial y_i} &= -\sum_{j} p_{ij} {1\over w_{ij}}{\partial w_{ij} \over \partial y_i} + {1\over Z} \sum_j {\partial w_{ij} \over \partial y_i}\\ &=  -2\sum_{j} p_{ij} {w_{ij}}{(y_i-y_j)} + {2\over Z} \sum_j w_{ij}^2(y_i-y_j)  \end{gather}
$$

$$
{\partial w_{ij} \over \partial y_i} = 2\bigg({1\over 1 +\|y_i-y_j\|^2}\bigg)^2 (y_i-y_j) = 2w_{ij}^2(y_i-y_j)
$$

The $2$ can be adjusted in the learning rate hence 

$$
\begin{gather}{\partial\mathcal{L} \over \partial y_i}\sim  -\sum_{j} p_{ij} {w_{ij}}{(y_i-y_j)} + {1\over Z} \sum_j w_{ij}^2(y_i-y_j)  \end{gather}
$$
