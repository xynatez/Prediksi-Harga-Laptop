# Neural Network for Price Prediction

This Python script implements a simple neural network using a Multilayer Perceptron (MLP) to predict the price of a laptop based on various factors. The neural network includes functions for sigmoid activation, its derivative, feedforward, and feedback.

## Functions

- **sigmoid(x):** The sigmoid activation function.
- **sigmoid_derivative(x):** The derivative of the sigmoid function.
- **feedforward(X, W1, B1, W2, B2):** Performs the feedforward process for the neural network.
- **feedback(X, Y, Z, target, W2, learning_rate, W1):** Implements the feedback mechanism to update weights based on the error.
  
## Usage

1. Input your data (X), including factors like technical specifications, premium design and material, storage type, and additional features.
2. Set the initial weights (W1, W2) and biases (B1, B2).
3. Specify the target value (target) and learning rate (learning_rate).
4. Run the feedforward process to predict the price (Y).
5. Use the feedback mechanism to update weights based on the prediction error.
6. Display the predicted price and a conclusion based on a threshold (e.g., if predicted price > 0.5, consider the laptop expensive).

Feel free to modify the initial data, weights, biases, target, and learning rate for different scenarios. The script provides a basic example of using a neural network for price prediction.
