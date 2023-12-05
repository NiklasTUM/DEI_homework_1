#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        # Calculate the prediction
        scores = np.dot(self.W, x_i)
        y_pred = np.argmax(scores)

        # Check if the prediction is incorrect (misclassified)
        if y_i != y_pred:
            # Update the weights based on the perceptron learning rule
            self.W[y_i, :] += x_i
            self.W[y_pred, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        self.W = self.W + learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)
        


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.hidden_size = hidden_size
        self.W1 = np.random.normal(0.1, 0.12, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,))
        self.W2 = np.random.normal(0.1, 0.12, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes,))

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def softmax(self, x):
        # Softmax activation function
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(self, X):
        # Compute the forward pass of the network.
        hidden_activations = self.relu(np.dot(X, self.W1.T) + self.b1)
        scores = np.dot(hidden_activations, self.W2.T) + self.b2
        return np.argmax(self.softmax(scores), axis=1)
    
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def compute_loss_and_gradients(self, X, y):
        # Compute the loss and gradients using backpropagation.

        # Forward pass
        hidden_activations = self.relu(np.dot(X, self.W1.T) + self.b1)
        scores = np.dot(hidden_activations, self.W2.T) + self.b2
        softmax_scores = self.softmax(scores)

        # Compute cross-entropy loss
        num_examples = X.shape[0]
        correct_class_scores = softmax_scores[range(num_examples), y]
        loss = -np.sum(np.log(correct_class_scores)) / num_examples

        # Backward pass
        dscores = softmax_scores.copy()
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        dW2 = np.dot(dscores.T, hidden_activations)
        db2 = np.sum(dscores, axis=0)

        dhidden = np.dot(dscores, self.W2)
        dhidden[hidden_activations <= 0] = 0  # ReLU backpropagation

        dW1 = np.dot(dhidden.T, X)
        db1 = np.sum(dhidden, axis=0)

        return loss, dW1, db1, dW2, db2

    def train_epoch(self, X, y, learning_rate=0.001):
        # Train the model for one epoch using stochastic gradient descent.
        num_examples = X.shape[0]
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        total_loss = 0
        for i in indices:
            x_i = X[i, :]
            y_i = y[i]

            # Compute loss and gradients
            loss, dW1, db1, dW2, db2 = self.compute_loss_and_gradients(x_i.reshape(1, -1), np.array([y_i]))

            # Update parameters
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            total_loss += loss

        return total_loss / num_examples


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
