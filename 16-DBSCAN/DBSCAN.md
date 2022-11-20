# DBSCAN

Density-Based Spatial Clustering of Applications with Noise(DBSCAN) is a unsupervised clustering algorithm. It clusters data on the basis of distance.

DBSCAN defines two parameters. First $\epsilon$, which helps in defining the epsilon neighbours of a point $p$ denoted as $\ce{N_\epsilon}(p)$.  

$$
\ce{N_\epsilon}(p)= \{q \in \mathcal{D}\mid \mathrm{dist}(p,q) \le \epsilon\}
$$

where $q$ is another point in the data set $\mathcal{D}$ and $\mathrm{dist}(p,q)$ is the distance between the two points. So all the points which satisfies the above condition belongs to the epsilon neighbour of the point $p$

The other parameter is $\mathrm{MinPts}$ which helps in distinguishing the core points and the boundary points as shown in the figure. A point $p$ is called as core point with respect to the parameter $\epsilon$ if 

$$
\# \ce{N_\epsilon }(p) \ge \mathrm{MinPts} 
$$

![Untitled](DBSCAN%20cdcd64219ce9440f949646778eb9e28a/Untitled.png)

### Definitions

A point $q$ is *directly density reachable* from a point $p$ with respect to $\epsilon$ and $\ce{MinPts}$ is  

$$
q \in \ce{N_\epsilon }(p)\\
\#\ce{N_\epsilon }(p) \ge \ce{MinPts} 
$$

![Untitled](DBSCAN%20cdcd64219ce9440f949646778eb9e28a/Untitled%201.png)

A point $q$ is *density reachable* from a point $p$ with respect to $\epsilon$ and $\ce{MinPts}$ if there is a chain of points $q= q_1,q_2,\cdots ,q_n=p$ exists such that $q_{i+1}$ is directly density reachable to $q_i$

![Untitled](DBSCAN%20cdcd64219ce9440f949646778eb9e28a/Untitled%202.png)

<aside>
📌 *density reachable* is a generalisation of *directly density reachable*. But for these two $q$ should be core point.

</aside>

A point $p$ is *density connected* to a point $q$ with respect to $\epsilon$ and $\ce{MinPts}$ if there is a point $o$ such that both $p$ and $q$ are density reachable from $o$

![Untitled](DBSCAN%20cdcd64219ce9440f949646778eb9e28a/Untitled%203.png)

So we define a *cluster $C$* if it satisfies the following condition

1. $\forall p,q:$ if $p \in C$ and $q$ is density reachable from point $p$ with respect to $\epsilon$  and $\ce{MinPts}$ then $q \in C$
2. $\forall p,q \in C$ then $p$ will be density connected to $q$ with respect to $\epsilon$  and $\ce{MinPts}$

Let $C_1,\cdots, C_k$ be the clusters of the data set $\mathcal{D}$ with respect to $\epsilon$ and $\ce{MinPts}$. Then we define *noise* as the set of points not belonging to any cluster $C_i$ is noise.

So the algorithm for clustering the data goes as follows 

1. First using our parameters we identify all the core points
2. Then we iterate to all the core points. If a core point is doesn’t belong to any cluster, it is assigned a new category. Moreover, all the core points belonging to its $\epsilon$ Neighbourhood are assigned to the same cluster
3. Now we iterate to all the non-core points. If there is a core point present in its $\epsilon$ Neighbourhood then we assign it to the same cluster otherwise we define it as noise.

### Questions

1. **How does $\epsilon$ changes the cluster ?**
    
    If $\epsilon$ has a very small value then clusters should be dense otherwise they would be considered as noise. If $\epsilon$ has a very large value then their is a possibility that two close enough clusters gets combined into a single one.
    
2. **How to get the optimum parameters for DBSCAN ?**
    
    There is no perfect algorithm of finding the best value for $\ce{MinPts}$. Now for each point we find the average distance for the first $k$ nearest points where $k= \ce{MinPts}$.  We plot the graph in the increasing order of the average values i.e. the x-axis is just the index of the point and y-axis is their respective average distance. We take the that $\epsilon$ value were the curve starts to bend. (like the elbow curve in K-means).
    

![Untitled](DBSCAN%20cdcd64219ce9440f949646778eb9e28a/Untitled%204.png)