# Introduction to Neural Networks

Neural Networks contains an input layer, one or more hidden layers, and an output layer. Each node, connects to another and has an associated weight and bias. 

![Red → Input features
Blue →Hidden layers
Green → Output](Introduction%20to%20Neural%20Networks%201d003390c7824c87b65c83cc4be8ec64/Untitled.svg)

Red → Input features
Blue →Hidden layers
Green → Output

So each node has its own weights and bias i.e. there is linear regression going on for each node. 

But linear functions composing linear functions will result in another linear function. Hence the hidden layers won’t be much help, they will perform similar to linear regression. So we use a activation function. 

Let the Input features i.e. our data set be $X \in \R^{n}$. Then we take the first weight matrix $W_1 \in \R^{n\times k}$ and the bias vector $b_1\in \R^k$ which will give 

$$
Z_1 = W_1X + b_1 
$$

then we take the activation function $f$ such that 

$$
A_1 = f(Z_1)
$$

Now this $A_1$ will be our first hidden layer. Now we will again do the same procedure 

$$
Z_2 = W_2A_1 + b_2\\
A_2 = f(Z_2)
$$

Let take only one hidden layer, then $A_2$ will be our output. This way we can build non-linear decision boundaries.

Then we can define our loss function $\mathcal{L}$ as the negative logarithm of the distance of our predicted output from the actual output. 

Now we can apply back propagation i.e. by find the partial derivative of our lost function with respect to these parameters.

$$
W_i =W_i - \alpha {\partial \mathcal{L}\over \partial W_i}\\
b_i =b_i - \alpha {\partial \mathcal{L}\over \partial b_i}\\
\forall i=\{1,2\}
$$

where $\alpha$ is our learning rate. Now, more the layers we add more will be our weight matrices.

 Below is the code for the Neural Networks.

[Machine-Learning/Neural Networks on MNIST.ipynb at main · Divyanshu-Bhatt/Machine-Learning](https://github.com/Divyanshu-Bhatt/Machine-Learning/blob/main/Neural%20Networks%20on%20MNIST.ipynb)

### Questions

1. **Examples of some of the activation functions ?**
    
    
    | Names | Function |
    | --- | --- |
    | Sigmoid | ⁍ |
    | ReLU | ⁍ |
    | tanh | ⁍ |
2. **Difference between Forward Propagation and Backward Propagation ?**
    
    Forward propagation is when the data set is fed in the forward direction through the network i.e. using the weights, biases and the activation function we get our output. Whereas, Back propagation is when the output will help to change the weights and biases with the help of the loss function.