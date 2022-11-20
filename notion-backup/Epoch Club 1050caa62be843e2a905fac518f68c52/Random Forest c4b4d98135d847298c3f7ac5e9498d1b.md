# Random Forest

It is also a type of supervised learning and based on decision trees. Random forest is based on bagging method. 

The original data set $\mathcal{D}$ is divided into $N$ equal smaller data sets $d_1,d_2\cdots d_N$. For each data set a decision tree is trained. During testing, the data point is given to each decision tree. All the decision tress will give there respective prediction and the category chose by the majority of the trees is considered as the category of the data point.   

![random.svg](Random%20Forest%20c4b4d98135d847298c3f7ac5e9498d1b/random.svg)

### Questions

1. **What is Ensemble Learning ?**
    
    Ensemble learning is a method in which different models are combined to produce an effective optimal prediction model like Random Forest algorithm.
    
2. **What is Bagging ?**
    
    In bagging, a random sample of data in a training set is selected with replacement. Models are trained on these sample of data and then the average or majority of those predictions yield a more accurate estimate.
    
3. **How is  Random Forest better than Decision Tree ?**
    
    Decision tree generally results in an overfitting model unlike Random Forest.