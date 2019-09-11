"""
trainer for IMDB classifier
Created by Chenyang, Sept. 2019

Learning rate: 0.1 (Total loss = average of per sample loss in the batch)
Learning rate decay: None (fixed learning rate)
Batch size: 20
Regularization: None
Number of epochs: 300

For epochs = 1, .., 300:
|	For batches of the training set:
|	|	Compute gradient
|	|	Update parameters
|	Predict on the validation set
Choose the best validation model
Predict on the test set
"""

import pickle as pkl
from model.logistic_classifier import LogisticClassifier
from build_imdb_dataset import VOCAB_SIZE
from utils.data_loader import DataLoader
from utils.ce_loss import CrossEntropyLoss
from optimizer.gradient_decent import GradientDecent

MAX_EPOCH = 300
LEARNING_RATE = 0.01
BATCH_SIZE = 20

with open('imdb_data.pkl', 'br') as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pkl.load(f)

model = LogisticClassifier(dim=VOCAB_SIZE)
loss_criterion = CrossEntropyLoss()
optimizer = GradientDecent(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(X_train, y_train, bs=BATCH_SIZE)
val_loader = DataLoader(X_val, y_val, bs=BATCH_SIZE)
test_loader = DataLoader(X_val, y_val, bs=BATCH_SIZE)


for epoch in range(MAX_EPOCH):
    train_loss = 0
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred) / len(X)
        print(loss)
        optimizer.step(loss)
        train_loss += loss
    print(f'Training loss is {train_loss}')

    val_loss = 0
    for X, y in val_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred)
        val_loss += loss
    print(f'Validation loss is {val_loss}')

