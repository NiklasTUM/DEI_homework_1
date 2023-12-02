#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import utils


# Q2.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        return self.linear(x)


# Q2.2
class FeedforwardNetwork(nn.Module):
    def __init__(self, n_classes, n_features, hidden_size, layers, activation_type, dropout):
        super(FeedforwardNetwork, self).__init__()

        # Choose the activation function
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

        # Define the layers
        self.layers = nn.ModuleList()
        input_size = n_features

        # Add hidden layers
        for _ in range(layers):
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # Pass the input through each layer
        for layer in self.layers:
            x = layer(x)
        # Output layer
        x = self.output_layer(x)
        return x


def train_batch(X, y, model, optimizer, criterion, device):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, device):
    X = X.to(device)
    scores = model(X)
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels.cpu()


@torch.no_grad()
def evaluate(model, X, y, criterion, device):
    model.eval()
    X, y = X.to(device), y.to(device)
    logits = model(X)
    loss = criterion(logits, y)
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss.item(), n_correct / n_possible


def plot(epochs, plottables, name='', ylim=None):
    """Plot the plottables over the epochs.

    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main(chosen_model='logistic_regression', epoch_num=20, batch_size=16, learning_rate=0.1, l2_decay=0,
         hidden_size=200, layers=2, dropout=0.0, activation='relu', optimizer_strategy='sgd'):
    # Rest of the setup remains the same

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats = dataset.X.shape[1]

    learning_rates = [0.1, 0.01, 0.001]

    results = {"train_losses_0.1": [], "valid_losses_0.1": [], "valid_accs_0.1": [],
               "train_losses_0.01": [], "valid_losses_0.01": [], "valid_accs_0.01": [],
               "train_losses_0.001": [], "valid_losses_0.001": [], "valid_accs_0.001": [],
               "best_lr": 0, "best_valid_acc": 0, "best_model": None,
               "test_accuracy": []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    for lr in learning_rates:
        start_time = time.time()
        print(f"Training with learning rate: {lr}")

        # Initialize the model and move it to the GPU if available

        if chosen_model == 'logistic_regression':
            model = LogisticRegression(n_classes, n_feats).to(device)
        else:
            model = FeedforwardNetwork(
                n_classes,
                n_feats,
                hidden_size,
                layers,
                activation,
                dropout
            ).to(device)

        # Get an optimizer
        optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

        optim_cls = optims[optimizer_strategy]
        optimizer = optim_cls(
            model.parameters(),
            lr=lr,
            weight_decay=l2_decay)

        # training loop
        epochs = torch.arange(1, epoch_num + 1)

        for epoch in epochs:
            print('Training epoch {}'.format(epoch))
            epoch_train_losses = []
            for X_batch, y_batch in train_dataloader:
                loss = train_batch(
                    X_batch, y_batch, model, optimizer, criterion, device)
                epoch_train_losses.append(loss)

            epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
            val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion, device)

            print('Training loss: %.4f' % epoch_train_loss)
            print('Valid acc: %.4f' % val_acc)

            results["train_losses_" + str(lr)].append(epoch_train_loss)
            results["valid_losses_" + str(lr)].append(val_loss)
            results["valid_accs_" + str(lr)].append(val_acc)

            # save the best validation accuracy and learning rate
            if val_acc > results["best_valid_acc"]:
                results["best_valid_acc"] = val_acc
                results["best_lr"] = lr

        # save the best model
        if results["best_lr"] == lr:
            results["best_model"] = model

        _, test_acc = evaluate(model, test_X, test_y, criterion, device)
        results["test_accuracy"].append(test_acc)
        print('Final Test acc: %.4f' % test_acc)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    print("Test accuracies: ")
    print(results["test_accuracy"])

    best_lr = results["best_lr"]
    print('Best learning rate: %f' % best_lr)
    # plot
    if chosen_model == "logistic_regression":
        config = (
            f"batch-{batch_size}-lr-{best_lr}-epochs-{epoch_num}-"
            f"l2-{l2_decay}-opt-{optimizer_strategy}"
        )
    else:
        config = (
            f"batch-{batch_size}-lr-{best_lr}-epochs-{epoch_num}-"
            f"hidden-{hidden_size}-dropout-{dropout}-l2-{l2_decay}-"
            f"layers-{layers}-act-{activation}-opt-{optimizer_strategy}"
        )

    losses = {
        "Train Loss": results["train_losses_" + str(best_lr)],
        "Valid Loss": results["valid_losses_" + str(best_lr)],
    }

    print(losses["Train Loss"])
    print(losses["Valid Loss"])

    valid_accs = results["valid_accs_" + str(best_lr)]
    print("Final validation accuracy: " + str(valid_accs[-1]))
    # Choose ylim based on model since logistic regression has higher loss
    if chosen_model == "logistic_regression":
        ylim = (0., 1.6)
    elif chosen_model == "mlp":
        ylim = (0., 1.2)
    else:
        raise ValueError(f"Unknown model {chosen_model}")
    plot(epochs, losses, name=f'{chosen_model}-training-loss-{config}', ylim=ylim)
    accuracy = {"Valid Accuracy": valid_accs}
    plot(epochs, accuracy, name=f'{chosen_model}-validation-accuracy-{config}', ylim=(0., 1.))


main()