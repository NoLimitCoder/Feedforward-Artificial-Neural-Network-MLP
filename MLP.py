import numpy as np

# Define the neural network
class mlp():
    def __init__(self, lr, num_in, num_hidden, num_out):
        self.lr = lr
        self.num_hidden = num_hidden
        self.weights_in = np.random.random_sample((num_in + 1, self.num_hidden))  # +1 for bias
        self.weights_out = np.random.random_sample((self.num_hidden + 1, num_out))  # +1 for bias

    def activation(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid activation function

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def act_der(self, x):
        return x * (1 - x)  # Derivative of the sigmoid function

    def train(self, X, t, epochs=1000):
        errors = []
        for epoch in range(epochs):
            error = self.train_single_epoch(X, t)
            errors.append(error)
            if (epoch + 1) % 1000 == 0:
                predictions = self.predict(X)
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(t, axis=1))
                print(f"Epoch {epoch + 1}: Loss = {error}, Accuracy = {accuracy}")
        return errors

    def train_single_epoch(self, X, t):
        # Add bias to input
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        # Forward pass
        z = np.dot(X_with_bias, self.weights_in)
        h = self.activation(z)
        # Add bias to hidden layer
        h_with_bias = np.hstack((h, np.ones((h.shape[0], 1))))
        y = np.dot(h_with_bias, self.weights_out)
        predictions = self.activation(y)

        # Error calculation
        cost = predictions - t
        error = 0.5 * np.mean(cost ** 2)

        # Backpropagation
        # Output layer gradients
        delta_out = self.act_der(predictions)
        dcost_dout = cost * delta_out
        dcost_weights_out = np.dot(h_with_bias.T, dcost_dout)

        # Hidden layer gradients
        dcost_dah = np.dot(dcost_dout, self.weights_out[:-1].T)  # Exclude bias weights
        din_dhidden = self.act_der(h)
        dcost_weights_hidden = np.dot(X_with_bias.T, din_dhidden * dcost_dah)

        # Weight updates
        self.weights_in -= self.lr * dcost_weights_hidden
        self.weights_out -= self.lr * dcost_weights_out

        return error

    def predict(self, X):
        # Add bias to input
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        hidden_in = np.dot(X_with_bias, self.weights_in)
        hidden_act = self.activation(hidden_in)
        # Add bias to hidden layer
        hidden_act_with_bias = np.hstack((hidden_act, np.ones((hidden_act.shape[0], 1))))
        out_in = np.dot(hidden_act_with_bias, self.weights_out)

        return self.softmax(out_in)

