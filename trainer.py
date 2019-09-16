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
from copy import deepcopy
import numpy as np
from tqdm import tqdm

MAX_EPOCH = 300
LEARNING_RATE = 0.1
BATCH_SIZE = 20

with open('imdb_data.pkl', 'br') as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pkl.load(f)

model = LogisticClassifier(dim=VOCAB_SIZE)
loss_criterion = CrossEntropyLoss(reduce=False)

train_loader = DataLoader(X_train, y_train,  bs=BATCH_SIZE)
val_loader = DataLoader(X_val, y_val, bs=BATCH_SIZE)
test_loader = DataLoader(X_val, y_val, bs=BATCH_SIZE)


best_model = None
best_val_loss = 9999
train_val_loss_list = []
best_epoch_num = None
for epoch in tqdm(range(MAX_EPOCH)):
    train_loss = 0
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred)
        model.gradient_decent_step(X, y, y_pred)
        train_loss += loss
    # print(f'Training loss is {train_loss/len(train_loader)}')

    val_loss = 0
    for X, y in val_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred)
        val_loss += loss
    # print(f'Validation loss is {val_loss/len(val_loader)}')

    # for the training curve
    train_val_loss_list.append((train_loss/len(train_loader), val_loss/len(val_loader)))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch_num = epoch + 1
        if best_model is not None:  # collect garbage
            del best_model
        best_model = deepcopy(model)

print(f'Best val loss at: {best_epoch_num:d} th epoch')
# save train and val loss to pickle
with open('train_val_loss.pkl', 'bw') as f:
    pkl.dump(train_val_loss_list, f)

# testing
del model
model = best_model
gold_test_list = []
pred_test_list = []
for X, y in val_loader:
    y_pred = model(X)
    gold_test_list.append(np.asarray(y))
    pred_test_list.append(np.asarray(y_pred))

gold_test_list = np.concatenate(gold_test_list)
pred_test_list = np.concatenate(pred_test_list)
pred_test_list[pred_test_list >= 0.5] = 1
pred_test_list[pred_test_list < 0.5] = 0

acc = sum(pred_test_list == gold_test_list) / len(pred_test_list)
print(f'Test accuracy is: {acc:2f}')
