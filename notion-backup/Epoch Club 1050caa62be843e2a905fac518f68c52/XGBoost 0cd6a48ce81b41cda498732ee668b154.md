# XGBoost

This is an extension to the Gradient Boost technique. 

Let our data $\mathcal{D} = \{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$ where $x_i \in \R^n$ and $y_i \in \R$ or $y_i \in \{0,1\}$ based on our type of problem. 

Let $\mathcal{L}(y_i,f(x_i))$ be the loss function.We add a regulariser to the Gradient Boost Machine and XGBoost is when we are finding the appropriate value for each of the leaf nodes. Earlier we use to do it as following

$$
\gamma_j = \argmin_\gamma \sum_{x_i \in l_j}\mathcal{L}(y_i,f_m(x_i)+\gamma)
$$

but now we will add a regulariser term i.e. 

$$
\gamma_j = \argmin_\gamma \sum_{x_i \in l_j}\mathcal{L}(y_i,f_m(x_i)+\gamma)  +{1\over2}\lambda \gamma^2
$$

where $\lambda$ is our regulariser parameter. We can solve it using Taylor series 

Moreover we can  remove the constant function $\mathcal{L}(y_i,f_m(x_i))$ so 

$$
L = \sum_{x_i \in l_j}\bigg( {d\mathcal{L}(y_i,f_m(x_i)  )\over d f_m}\gamma+{1\over2}{d^2\mathcal{L}(y_i,f_m(x_i))\over df_m^2}\gamma^2\bigg)+{1\over2}\lambda \gamma^2
$$

Finding the value of $\gamma_j$ using differentiation 

$$
\sum_{x_i \in l_i}{d\mathcal{L}(y_i,f_m(x_i)  )\over d f_m}+\sum_{x_i \in l_i}{d^2\mathcal{L}(y_i,f_m(x_i))\over df_m^2}\gamma + \lambda \gamma =0\\
\gamma_j = {\displaystyle\sum_{x_i\in l_i}{-d\over df_m}\mathcal{L}(y_i,f_m(x_i))\over \displaystyle\sum_{x_i\in l_i}{-d^2\over df_m^2}\mathcal{L}(y_i,f_m(x_i))+\lambda}
$$

Now substituting the above $\gamma_j$ in $L$ 

$$
L  = {1\over 2}{\displaystyle\bigg(\sum_{x_i\in l_i}{-d\over df_m}\mathcal{L}(y_i,f_m(x_i))\bigg)^2\over \displaystyle\sum_{x_i\in l_i}{-d^2\over df_m^2}\mathcal{L}(y_i,f_m(x_i))+\lambda}
$$

The above is defined as the similarity score. It helps in measuring the quality of the tree. As similarity score would be relative to avoid more computation the half in front of the whole function is removed hence 

$$
\ce{Similarity Score} = {\displaystyle\bigg(\sum_{x_i\in l_i}{-d\over df_m}\mathcal{L}(y_i,f_m(x_i))\bigg)^2\over \displaystyle\sum_{x_i\in l_i}{-d^2\over df_m^2}\mathcal{L}(y_i,f_m(x_i))+\lambda}
$$

Another term called Gain is defined (like information gain) which tells how much good is our split is. It is calculated as the following 

$$
\ce{Gain} = \ce{Leftchild}_\ce{Similarity Score}+\ce{Rightchild}_\ce{Similarity Score}-\ce{Root}_\ce{Similarity Score} 
$$

Now the above decision tree will be created which would be added to the previous function like done in GBM i.e. 

$$
f_{m+1}(x) = f_m(x) + \eta\gamma
$$

### Questions

1. **How is XGBoost better than Gradient Boost Machine ?**
    
    We don’t have to apply gradient descent for each smaller model resulting in less time in training. Moreover, we can easily use a regulariser resulting in better predictions also.
    
2. **How to prune in XGBoost ?**
    
    XGBoost starts pruning trees backward by defining a parameter $\theta$. Until we find any node such that $\ce{Gain} - \theta > 0$, the nodes are pruned.