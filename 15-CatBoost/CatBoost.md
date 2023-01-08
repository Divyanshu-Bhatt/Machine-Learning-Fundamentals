# CatBoost

Categorical Boosting is an extension of Gradient Boosting Machine. It used binary decision trees as its weak models. The main thing CatBoost does is to transform categorical features into numerical features

Categorical features are the features that can take one of a limited number of possible values. These values are usually fixed. Examples favourite primary colour (blue, green, red) etc.

One way is converting the categorical value to one hot encoded vectors. Example, the left column would be turned to right set of columns

| Favourite Primary Colour |
| --- |
| Red |
| Blue |
| Blue |
| Green |
| Red |
| Green |

| Red | Blue | Green |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |

Another way is by first permuting the set of input objects in a random order. Now we transform the categorical features to numerical values by the following formula

$$
\ce{value} = {\ce{Count in Class} + \ce{prior} \over \ce{total Count} + 1}
$$

Where $\ce{value}$ is the value in the new column. $\ce{prior}$ is a pre-defined parameter by user. $\ce{Count in Class}$ is the total number of objects in the current training data set with the current categorical feature value. $\ce{total Count}$is the total number of objects (up to the current one) that have a categorical feature value matching the current one.

This will help in converting the data from the above one to the below one (let the left one be the table we get after permutation). (Let $\ce{prior}=0.1$). 

| Favourite Primary Colour | Target Values |
| --- | --- |
| Red | 1 |
| Blue | 0 |
| Blue | 1 |
| Green | 1 |
| Red | 0 |
| Green | 1 |

| Count in Class | Total Counts | Favourite Primary Colour | Target Values |
| --- | --- | --- | --- |
| 0 | 0 | 0.1 | 1 |
| 0 | 0 | 0.1 | 0 |
| 0 | 1 | 0.05 | 1 |
| 0 | 0 | 0.1 | 1 |
| 0 | 1 | 0.05 | 0 |
| 1 | 1 | 0.55 | 1 |

The columns $\ce{total counts}$ and $\ce{Count in Class}$ are not there in  the actual algorithm.

Then the model proceeds by building symmetric binary trees for each permutation of the data. Symmetric binary trees are the trees in which at each level, all the nodes present at that level will have the same split condition

<p align="center">
     <img src="https://github.com/Divyanshu-Bhatt/Machine-Learning-Fundamentals/blob/main/15-CatBoost/images/symmetric_tree.svg" width="300"/>
</p>

### Questions

1. **How are trees built different in CatBoost as compared to GBM ?**
    
    In CatBoost the trees are symmetrical in nature i.e. node at each level has the same condition for splitting.
