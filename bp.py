import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))

    def backward(self, X, y, learning_rate):
        error = y - self.output
        delta_output = error * sigmoid_derivative(self.output)
        error_hidden = delta_output.dot(self.weights2.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden)
        self.weights2 += self.hidden.T.dot(delta_output) * learning_rate
        self.weights1 += X.T.dot(delta_hidden) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        self.forward(X)
        return self.output

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(2, 2, 1)

nn.train(X, y, epochs=10000, learning_rate=0.1)

print(nn.predict(X))
