# Decision Tree

Decision Tree is a type of supervised learning in which we are given some features using them we have to find their respective category. Example, finding the type of flower given different features such as sepal length, sepal width, petal length etc

Let our training set be $\mathcal{D} = \{(x_1,y_1),(x_2,y_2) \cdots (x_m,y_m)\}$ where $x_m \in \mathbb{R}^n$ are the features and $y_m \in \{1,2\cdots,k\}$ are our target values.

Solving with the given below example. Below are three possible classes i.e. $y \in \{1,2,3\}$. Each small colour ball represents a category. Blue nodes means its a decision node i.e. the data in that node doesn’t belong to a single class or we need to further segregate the data . White nodes means its a leaf node i.e. there is no further branches present.

![tree.png](03-Decision%20Tree/tree.png)

So we need to find parameters such that we can make some useful splits with the given data. So defining a term called entropy which helps us with it.

Entropy$(E)$ tells us the amount of uncertainty or randomness in the data or the purity of that node i.e.    

$$
E = -\sum_{i=1}^k p(y= i)\log_k p(y=i)
$$

where $p(y=i) = {\ce{number of data points of i^{th}class } \over \ce{total number of data points}}$ 

$E = 1$ means it has max uncertainty in the class or perfectly impure data and $E = 0$ means no randomness i.e. there is one class present. According to the example uncertainty of level $0$ node (top most node) is $1$ because its the most uncertain state. 

Defining another term called as information gain which tells us which parameter makes the better split. The parameter which maximises the information gain$(IG)$ is taken

$$
{IG} = E(parent) - \sum_{i=1}^k w_iE(child_i) 
$$

where $E(parent)$ is entropy of the parent node, $E(child_i)$ is the entropy of the $i^{th}$ child $w_i$ is the percentage of members present in the $i^{th}$ node as compared to the parent node.

So our model runs for every possible split for a given feature and finds the best parameter which maximises the information gain.

### Questions

1. **Which nodes are considered as pure ?**
    
    The node with Entropy $=0$ is called pure node i.e. there is only single class present at that node hence no subdivision will be needed. 
    
2. **What is the difference between pure node and leaf node ?**
    
    All pure nodes will be leaf nodes but all leaf nodes won’t be pure nodes. Leaf nodes are the nodes where we don’t further divide the tree because it may cause overfitting or it has become a pure node.
    
3. **How to stop the model from overfitting from decision tree ?**
    
    We can introduce a stopping criteria i.e. we predefine the maximum depth of the tree possible otherwise the decision tree algorithm will overfit the data.
    
4. **What is entropy ?**
    
    Entropy tells us the randomness in the data points present at that node i.e. more the data points of different category the harder will be to find decide the class of a specific example at that node. Hence we want to decrease entropy.
    
5. **What does information gain represents ?**
    
    Information gain shows the change in entropy from the previous state to the new state. As we want to decrease the entropy as much as possible hence we take the maximum information gain.
