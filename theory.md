# Theory

A brief discussion of my understanding of the theory behind a Neural Network, along with decisions made when implementing the code, and necessary transformations to prepare the data for training/querying.

## How a Neural Network works

A Neural Network of depth n is an n-partite graph (generalisation of a bipartite graph). There are input nodes at one end, hidden nodes in between, and output nodes at the other end. Signals travel from the input nodes down to the output nodes, where the signal strength is altered by weights assigned to each edge. The signals into a node from each incoming edge are summed before being passed through an activation function. The purpose of the activation function is to trigger a release once a certain threshold is met. The signals arriving at the output nodes represent the decision made by the network.

Note that taking sums scaled by weights corresponds to matrix multiplication, where we put all the weights from a given layer into a matrix. Therefore, propagating the signal forward involves repeatedly doing matrix multiplication followed by applying the activation function.

The key to training the Neural Network is tuning the weights attached to each edge. This is done by calculcating the error. When training the model, we know the target output. Thus, we can get an error at each of the output nodes. Backpropagating the error is done by matrix multiplication with a transposed weight matrix. This way the error is split up based on the relevant weights.

How do we now update the weights? The newly calculcated error at each node (other than the inputs) is used as a guide. We want to find the weights which minimise the error, but doing this exactly is very messy algebraically. Instead numerical methods are appealed to - Gradient Descent.

### Gradient Descent

We want to minimise a complicated mathematical function. The idea is to start at a random point and move 'downwards' until one finds a valley/local minimum. This might not be the global minimum, but one can run the training several times to find an optimal start. In practice, this means updating the location with the following sequence:

new_x = old_x - a * grad(f)   :   where 'a' is a chosen 'learning rate' parameter, and f is the given function.

Given an appropriate learning rate, this sequence ought to tend toward a local minimum. Therefore, each time we apply this formula, the weights should update in such a way that it decreases the error. Rewriting the above formula in the neural network context yields:

new_weight(i,j) = old_weight(i,j) - a * d(Error)/d(weight(i,j))   :   where weight(i,j) denotes the weight associated to the edge connecting node i to node j in the next level.

We choose $(target - output)^2$ for the error function. This is always positive, but unlike the absolute value it is flatter near the minimum so we lower the risk of repeatedly overshooting the minimum (our error correction scales down as the gradient becomes flatter). Also, this function is differentiable everywhere (unlike absolute value).

What remains is to calculate grad(f). This depends on the choice of activation function.

Now we have a computationally efficient way of updating the weights, and our network is able to train itself.

## Implementing the NeuralNetwork Class

(1) Activation Function

The sigmoid function (1/(1 + exp(-x)) is chosen as the activation function. It looks like an 'S' shape. Its smooth behaviour mimics real biological neurons better than, say, the Heavyside step function (neural networks like smoothness, not jagged shapes). Also, being continuously differentiable, it works well with gradient descent. Finding the derivative comes down to differentiating $(output_j(weights(i,j)) - target_j)^2$ (noting that only edges ending at node j will contribute).

Recall: output_j(weights(i,j)) = sigmoid( weighted sum of previous outputs )

The derivative expression is not hard to calculate but a little messy to write (instead see neuralnetwork.py).

(2) Picking initial weights

Apparently, a sensible way of picking initial weights is sampling from a normal distribution of mean 0 and variance 1/sqrt(num_nodes_going_inward). This choice relates to the section below.

## Transforming the Data

### Inputs

The neural network's ability to learn depends on the gradient. A flatter gradient means less learning as the gradient descent updates are reduced. Looking at a plot of the sigmoid function, it is important to pick numbers not too large in absolute value, as the function approaches an asymptote in both directions. We also don't want zero values as they can kill the entire expression in the weight update formula (so no learning takes place). Rescaling the inputs to fit the range 0.01-1.00 seems a happy medium.

### Targets

Like with inputs, the targets must be scaled appropriately. We know that the image of the sigmoid is 0 to 1. Therefore, we have to scale the targets down to meet this range. Note that neither 0 nor 1 can ever be attained by the sigmoid (and if the neural network tried to, it would have to send the signal to +- infinity, blowing up the weights). Therefore, 0.01 to 0.99 is a nice choice.