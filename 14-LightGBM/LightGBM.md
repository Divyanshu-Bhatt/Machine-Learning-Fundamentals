# LightGBM

LightGBM is an extension of Gradient Boosting Machine which helps in reducing training time and reducing memory. It is a framework based on decision trees.

The main cost in GBDT (Gradient Boosted Decision Trees) lies is the learning of decision trees and specifically in choosing the best split points. GBDT performs the split on the data set as following.

Let a data set $\mathcal{D} = \{x_1,x_2,\cdots x_m\}$ where $x_i \in \R^n$. Let we have already defined a loss function $\mathcal{L}$ according to target values in our data set. In each iteration of gradient boosting let the negative gradients of the loss function with respect to the output of the model are denoted by $\{g_1,\cdots ,g_n\}$ i.e. 

$$
g_i = {-d\over df_m}\mathcal{L}(y_i,f_m(x_i))
$$

where $f_m$ is the output of the model

Now while training a decision tree it splits its node at the point where it has the largest information gain. For GBDT the information gain is measured using variance. 

Let $\mathcal{O}$ be the training data set at a fixed node of the decision tree. Then the variance split of the feature $j$ at point $d$ i.e. the condition $x_{ij} \le d$ belongs to the left child and $x_{ij} > d$ belongs to the right child, will be defined as  

$$
\mathbb{V}_{j\mid \mathcal{O}}[d] = {1\over n_\mathcal{O}}\bigg({(\sum_{x_i \in L} g_i)^2\over n_{Lj\mid \mathcal{O}} } + {(\sum_{x_i \in R} g_i)^2\over n_{Rj\mid \mathcal{O}}} \bigg)
$$

where  

$$
n_{\mathcal{O}} =\sum\mathcal{I}(x_i \in \mathcal{O})\\
n_{Lj\mid \mathcal{O}} = \sum\mathcal{I}(x_i \in \mathcal{O} , x_{ij} \le d)\\n_{Rj\mid \mathcal{O}} = \sum\mathcal{I}(x_i \in \mathcal{O} , x_{ij} > d)
$$

so we find $d^*$ such that 

$$
d^* = \argmax_d \mathbb{V}_{j \mid \mathcal{O}}[d]
$$

But this would cause a lot of computation. So LightGBM uses GOSS i.e. Gradient Boosted One Side Sampling. They estimate $\tilde{\mathbb{V}}_{j \mid\mathcal{O}}$ over a small subset of the $\mathcal{O}$. First they find training instances according to there absolute values of gradients in the descending order. Large gradient means large error i.e. the data point is not learned well.

Taking the first $a \times 100\%$ of the data in there subset $A$ then for the remaining set $A^c$ containing $(1-a)\times 100 \%$ instances of smaller gradients, we randomly choose a sample $B$ of size $b \times |A^c|$. Now the data is split according to the set $\mathcal{X} =A \cup B$.  

$$
\tilde{\mathbb{V}}_{j \mid \mathcal{O}}[d] = {1\over n_{\mathcal{X}}} \bigg({(\sum_{x_i \in A_l} g_i + {({1-a\over b})}\sum_{x_i \in B_l}g_i)^2\over n_{Lj\mid \mathcal{X}} } + {(\sum_{x_i \in A_r} g_i + {({1-a\over b})}\sum_{x_i \in B_r}g_i)^2\over n_{Rj\mid \mathcal{X}} } \bigg)
$$

where $A_l = \{x_i \in A, x_{ij} \le d\}$, $A_r = \{x_i \in A, x_{ij} > d\}$ and similarly for $B_l$ and $B_r$

The coefficient ${1-a \over b}$ is used to normalise the sum of the gradients over $B$ back to the size of $A^c$

The second thing that LightGBM uses is EFB i.e. Exclusive Feature Bundling. Many times in our data set their can exist two or more features such that they are rarely take nonzero values simultaneously. One-hot encoded features (helps in numerical representation of categorical features) are a perfect example of exclusive features. EFB bundles these features, reducing the dimensionality to improve efficiency while maintaining the accuracy.

### Questions

1. **How is LightGBM faster than GBM ?**
    
    Only a subset of the total data set is taken while calculating the variance. Thus it takes a less time in computation the variance making the algorithm fast.
    
2. **How are trees in LightGBM different then in GBM or XGBoost ?**
    
    While in other algorithms the trees goes level wise whereas in LightGBM it grows leaf wise
    

![Untitled](LightGBM%206808df6c53dd48b7b490a2423ee52b3c/Untitled.png)