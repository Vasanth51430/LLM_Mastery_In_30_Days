## **Forward and Backward Propagation**

In neural networks, **forward propagation** and **backward propagation** are the two fundamental processes involved in training. Here's a detailed breakdown of both, along with examples to clarify the concepts.

### 1. **Forward Propagation**:

Forward propagation is the process of passing the input data through the network layers, calculating the output (or prediction), and comparing it with the actual target to compute the loss.

#### Step-by-Step Process:

#### Step 1: **Input Layer**
- The process starts by feeding the input data into the network.
  
  **Example**: 
  Let's assume we have a neural network with two input features \(x_1 = 1\) and \(x_2 = 2\), and a target output \(y = 1\).

#### Step 2: **Weighted Sum (Linear Transformation)**
- Each input is multiplied by a corresponding weight and summed. Additionally, a bias term is added to the sum.
  
  Suppose we have two neurons in the hidden layer, and the corresponding weights and biases are:
  - \(w_1 = 0.4\), \(w_2 = 0.6\) (weights for first neuron)
  - \(w_3 = 0.5\), \(w_4 = 0.7\) (weights for second neuron)
  - Bias for the first neuron = \(b_1 = 0.2\)
  - Bias for the second neuron = \(b_2 = 0.3\)

  The weighted sum for the first hidden layer is calculated as:
  - Neuron 1: \(z_1 = (x_1 \times w_1) + (x_2 \times w_2) + b_1\)
  - Neuron 2: \(z_2 = (x_1 \times w_3) + (x_2 \times w_4) + b_2\)

  **Calculation**:
  - \(z_1 = (1 \times 0.4) + (2 \times 0.6) + 0.2 = 0.4 + 1.2 + 0.2 = 1.8\)
  - \(z_2 = (1 \times 0.5) + (2 \times 0.7) + 0.3 = 0.5 + 1.4 + 0.3 = 2.2\)

#### Step 3: **Activation Function**
- The output from each neuron is passed through an activation function (e.g., ReLU, sigmoid, tanh) to introduce non-linearity.
  
  Let's use the **ReLU** activation function: \(f(x) = \text{max}(0, x)\)

  **Calculation**:
  - For Neuron 1: \(a_1 = \text{ReLU}(z_1) = \text{max}(0, 1.8) = 1.8\)
  - For Neuron 2: \(a_2 = \text{ReLU}(z_2) = \text{max}(0, 2.2) = 2.2\)

#### Step 4: **Output Layer**
- The output from the hidden layer is fed to the output layer, where another set of weights and biases is applied.
  
  Let's assume we have the following weights and bias for the output neuron:
  - Weight for Neuron 1: \(w_5 = 0.8\)
  - Weight for Neuron 2: \(w_6 = 0.9\)
  - Bias for the output neuron: \(b_3 = 0.1\)

  The weighted sum for the output is:
  - \(z_{\text{output}} = (a_1 \times w_5) + (a_2 \times w_6) + b_3\)
  
  **Calculation**:
  - \(z_{\text{output}} = (1.8 \times 0.8) + (2.2 \times 0.9) + 0.1 = 1.44 + 1.98 + 0.1 = 3.52\)

#### Step 5: **Loss Calculation**
- The network's output is compared with the true target \(y\) using a loss function (e.g., mean squared error, cross-entropy loss).
  
  Let's use **mean squared error** (MSE) as the loss function:
  \[
  \text{Loss} = \frac{1}{2}(y_{\text{output}} - y_{\text{true}})^2
  \]

  **Calculation**:
  \[
  \text{Loss} = \frac{1}{2}(3.52 - 1)^2 = \frac{1}{2}(2.52)^2 = \frac{1}{2} \times 6.3504 = 3.1752
  \]

At this point, we have completed forward propagation.

### 2. **Backward Propagation**:

Backward propagation (or backprop) involves calculating the gradient of the loss function with respect to each weight in the network, and updating the weights using an optimization algorithm (e.g., gradient descent).

#### Step-by-Step Process:

#### Step 1: **Compute Loss Gradient (∂L/∂y)**
- First, we calculate the derivative of the loss function with respect to the output.
  
  \[
  \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = z_{\text{output}} - y_{\text{true}} = 3.52 - 1 = 2.52
  \]

#### Step 2: **Gradient at the Output Layer**
- Now, we calculate the gradients of the loss with respect to the weights and biases in the output layer.
  
  \[
  \frac{\partial \text{Loss}}{\partial w_5} = a_1 \times \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = 1.8 \times 2.52 = 4.536
  \]
  \[
  \frac{\partial \text{Loss}}{\partial w_6} = a_2 \times \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = 2.2 \times 2.52 = 5.544
  \]
  \[
  \frac{\partial \text{Loss}}{\partial b_3} = \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = 2.52
  \]

#### Step 3: **Backpropagate to Hidden Layer**
- Now we backpropagate the error to the hidden layer using the chain rule. The error signal at each hidden neuron is the product of the derivative of the activation function and the error propagated from the output layer.

  For ReLU, the derivative is:
  \[
  f'(z) = 
  \begin{cases} 
  1 & \text{if } z > 0 \\ 
  0 & \text{otherwise} 
  \end{cases}
  \]
  Since \(z_1 = 1.8 > 0\) and \(z_2 = 2.2 > 0\), both have derivatives of 1.

  - Error at Neuron 1: 
    \[
    \frac{\partial \text{Loss}}{\partial z_1} = w_5 \times \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = 0.8 \times 2.52 = 2.016
    \]
  - Error at Neuron 2: 
    \[
    \frac{\partial \text{Loss}}{\partial z_2} = w_6 \times \frac{\partial \text{Loss}}{\partial z_{\text{output}}} = 0.9 \times 2.52 = 2.268
    \]

#### Step 4: **Gradient at the Hidden Layer**
- Now calculate the gradients of the loss with respect to the weights and biases in the hidden layer.

  For Neuron 1:
  \[
  \frac{\partial \text{Loss}}{\partial w_1} = x_1 \times \frac{\partial \text{Loss}}{\partial z_1} = 1 \times 2.016 = 2.016
  \]
  \[
  \frac{\partial \text{Loss}}{\partial w_2} = x_2 \times \frac{\partial \text{Loss}}{\partial z_1} = 2 \times 2.016 = 4.032
  \]
  \[
  \frac{\partial \text{Loss}}{\partial b_1} = \frac{\partial \text{Loss}}{\partial z_1} = 2.016
  \]

  For Neuron 2:
  \[
  \frac{\partial \text{Loss}}{\partial w_3} = x_1 \times \frac{\partial \text{Loss}}{\partial z_2} = 1 \times 2.268 = 2.268
  \]
  \[
  \frac{\partial \text{Loss}}{\partial w_4} = x_2 \times \frac{\partial \text{Loss}}{\partial z_2} = 2 \times 2.268 = 4.536
  \]
  \[
  \frac{\partial \text{Loss}}{\partial b_2} = \frac{\partial \text{Loss}}{\partial z_2} = 2.268
  \]

#### Step 5: **Update Weights and Biases**
- Using gradient descent, we update each weight and bias by subtracting the gradient scaled by the learning rate (\(\eta\)).

  **Example (with \(\eta = 0.01\)):**

  - \(w_1 = w_1 - \eta \times \frac{\partial \text{Loss}}{\partial w_1} = 0.4 - 0.01 \times 2.016 = 0.37984\)
  - \(w_2 = 0.6 - 0.01 \times 4.032 = 0.55968\)
  - And similarly for other weights and biases.

This process is repeated for each training example across multiple epochs until the network converges (i.e., the loss becomes minimal).

---

### Summary:
- **Forward Propagation**: Involves calculating the weighted sums, applying activations, and computing the output and loss.
- **Backward Propagation**: Involves calculating the gradients of the loss with respect to the weights and biases, and updating them using gradient descent.

### Gradient Descent in Backward Propagation

In backward propagation, **gradient descent** is a key optimization algorithm used to minimize the loss function by updating the weights and biases of the neural network. The process is enhanced by various **optimizers**, which improve the efficiency and stability of the learning process.

Let's break it down step by step:

---

### **Gradient Descent: The Core Concept**

**Gradient Descent** is an iterative optimization algorithm used to minimize a loss function by adjusting the model's parameters (weights and biases). The main idea is to move towards the minimum of the loss function by taking small steps in the opposite direction of the gradient (the slope) of the loss with respect to the parameters.

#### Key Steps:
1. **Compute the Gradient**: 
   The gradient of the loss function with respect to each parameter (weight and bias) tells us the direction in which the loss function increases. We want to go in the opposite direction to minimize the loss.

   For a weight \( w \), the gradient is \( \frac{\partial \text{Loss}}{\partial w} \).

2. **Update the Parameters**:
   After calculating the gradient, the parameters are updated using the formula:
   \[
   w_{\text{new}} = w_{\text{old}} - \eta \times \frac{\partial \text{Loss}}{\partial w}
   \]
   where:
   - \( w_{\text{new}} \) is the updated weight.
   - \( \eta \) is the **learning rate**, a hyperparameter that controls how large each step is.
   - \( \frac{\partial \text{Loss}}{\partial w} \) is the gradient of the loss with respect to the weight.

#### Example:
- Suppose the weight \( w = 0.5 \), the gradient \( \frac{\partial \text{Loss}}{\partial w} = 0.2 \), and the learning rate \( \eta = 0.01 \).
  \[
  w_{\text{new}} = 0.5 - 0.01 \times 0.2 = 0.498
  \]
  The weight has been adjusted slightly in the direction that reduces the loss.

#### Types of Gradient Descent:
- **Batch Gradient Descent**: Uses the entire training dataset to compute the gradient in each step.
- **Stochastic Gradient Descent (SGD)**: Updates the weights after each training example (faster but noisier updates).
- **Mini-Batch Gradient Descent**: Combines both, where a small batch of data is used to compute the gradient.
