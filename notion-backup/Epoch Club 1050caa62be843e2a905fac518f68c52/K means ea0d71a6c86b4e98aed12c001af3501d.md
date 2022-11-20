# K means

$k$ means algorithm is an unsupervised learning algorithm. So the whole data set should be divided into $k$ different classes. Like kNN it is based on the principle that similar data perform clusters. 
$k$ means algorithm finds $k$ different clusters. Given the training set, $k$ different points are selected randomly which are considered as the centroids for each class. Now for the other data points in the training set, distance is calculated  from that point to all the centroids i.e.

$$
\|z-x_i\| = \bigg(\sum_{j=1}^n |z_j - x_{ij}|^p\bigg)^{1\over p}
$$

where $z$ is the other data points in the training set and $x_i$ is the coordinate of one of the centroids. The data point is given the same class as the class of the nearest centroid. 

After all the data points are classified to a class, the centroid of the each cluster is calculated by the following formula 

$$
{1\over b}\sum_{i=1}^b x_l
$$

where $x_l$ belongs to that particular class and $b$ are the total number of points present in that class. Again all the data points are again classified using these new centroids. The above process is repeated until there is no change in the centroids.    

<aside>
📌 As distance is the metric for the classification of a data point to a class, hence the data set is initially normalised to remove all the different scaling in different dimensions.

</aside>

[[Source](https://yihui.org/animation/example/kmeans-ani/)](https://assets.yihui.org/figures/animation/example/kmeans-ani/demo-c.mp4)

[Source](https://yihui.org/animation/example/kmeans-ani/)

As its not necessary that the random points we chose initially will result in good segregation of data into different classes. So we find the variation for each class and sum them. The iteration which has the least sum variation is taken as the best attempt for that particular $k$ value.

<aside>
📌 Coordinate-wise squared deviations from the centroid of the cluster of all the observations belonging to that cluster is called as the cluster variance

</aside>

As the $k$ value increases the variation decreases but the bias increases.

Now for finding the best value of $k$ we plot the graph of variance against the $k$ values.

![K means.drawio (1).svg](K%20means%20ea0d71a6c86b4e98aed12c001af3501d/K_means.drawio_(1).svg)

The optimal value of k is at the “elbow” i.e. the point after which the variance starts decreasing some what linearly. Thus for the above example the $k$ value is $4$

### Questions

1. **What is the difference between K means and KNN algorithm ?**
    
    Both K means and KNN uses distance metrics but K means is an unsupervised learning algorithm which finds cluster in the data whereas KNN is an supervised learning algorithm which finds the first $k$ closest data points to it and classify itself on the basis of their classes. 
    
2. **What is Elbow curve ?**
    
    It is a way of finding the best $k$ value for K means algorithm. When the variance graph starts decreasing linearly against values of $k$, its starting point is taken as the optimal value for $k$