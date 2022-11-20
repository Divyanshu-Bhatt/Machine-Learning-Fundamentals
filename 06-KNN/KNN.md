# k- Nearest Neighbour

It is a type of supervised learning algorithm. It tries to predict the correct class by calculating the distance between the new data point and all the other points.

The kNN algorithm is based on the fact that similar data perform clusters. Given a data set 
$\mathcal{D} = \{(x_1,y_1)\cdots(x_m,y_m)\}$ where $y = \{1,2,\cdots ,k\}$ be different classes.

![Knn.drawio (1).svg](k-%20Nearest%20Neighbour%2045d5857a088c44bd95f59aaa0bad793a/Knn.drawio_(1).svg)

In order to find the class of a new data point $z$, we find the distance of that data point from all the other points in the training data set i.e. 

$$
\|z-x_i\| \ \forall i \in 1,\cdots ,m
$$

<aside>
📌 As distance is the metric for the classification of a data point to a class, hence the data set is initially normalised to remove all the different scaling in different dimensions.

</aside>

Now we find the first $k$ closest point to $z$. The class with the most number of count in these $k$ closest point will be assigned to $z$. If there are two or more classes has the same and the maximum number of counts then either the data point is left unmarked or their is a coin flip (i.e. randomly assigned between the maximum count classes).

There are different types of distance metrics. The generalised distance metrics is 

$$
\|z-x_i\| = \bigg(\sum_{j=1}^n |z_j - x_{ij}|^p\bigg)^{1\over p}
$$

Now the most important thing is to find the optimal value of $k$. A small value for $k$ fits the data such that it has low bias but high variance. Graphically, the decision boundary will be more jagged. Whereas, large values of $k$ fits the data such that it has low variance but increased bias. Graphically, the decision boundary will be smoother.

There is no set way to find the value of $k$. One way is to train the model for different values of $k$, and take the value of $k$ which gives the least error.

### Questions

1. **When should KNN algorithm should be used ?**
    
    KNN algorithm is used when one requires high accuracy but that do not require a human-readable model.
    
2. **Explain one more different distance metrics  that can be used other than the above generalised one ?**
    
    Cosine distance can also be used as a distance metrics. It calculates the similarity between two vectors on the basis of the angle between them.