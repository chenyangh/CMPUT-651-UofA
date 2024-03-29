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

is_criterion_best_val_loss = False
best_model = None
best_val_loss = 9999
best_val_acc = 0
train_val_loss_list = []
train_val_acc_list = []
best_epoch_num = None
for epoch in tqdm(range(MAX_EPOCH)):
    train_loss = 0
    gold_train_list = []
    pred_train_list = []
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred)
        model.gradient_decent_step(X, y, y_pred)
        gold_train_list.append(np.asarray(y))
        pred_train_list.append(np.asarray(y_pred))
        train_loss += loss
    # print(f'Training loss is {train_loss/len(train_loader)}')
    gold_train_list = np.concatenate(gold_train_list)
    pred_train_list = np.concatenate(pred_train_list)
    pred_train_list[pred_train_list >= 0.5] = 1
    pred_train_list[pred_train_list < 0.5] = 0
    train_acc = sum(pred_train_list == gold_train_list) / len(pred_train_list)

    val_loss = 0
    gold_val_list = []
    pred_val_list = []
    for X, y in val_loader:
        y_pred = model(X)
        loss = loss_criterion(y, y_pred)
        gold_val_list.append(np.asarray(y))
        pred_val_list.append(np.asarray(y_pred))
        val_loss += loss
    # print(f'Validation loss is {val_loss/len(val_loader)}')
    gold_val_list = np.concatenate(gold_val_list)
    pred_val_list = np.concatenate(pred_val_list)
    pred_val_list[pred_val_list >= 0.5] = 1
    pred_val_list[pred_val_list < 0.5] = 0
    val_acc = sum(pred_val_list == gold_val_list) / len(pred_val_list)

    # for the training curve on loss
    train_val_loss_list.append((train_loss/len(train_loader), val_loss/len(val_loader)))
    train_val_acc_list.append((train_acc, val_acc))

    if is_criterion_best_val_loss:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_num = epoch + 1
            if best_model is not None:  # collect garbage
                del best_model
            best_model = deepcopy(model)
    else:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_num = epoch + 1
            if best_model is not None:  # collect garbage
                del best_model
            best_model = deepcopy(model)


print(f'Best model at: {best_epoch_num:d} th epoch, given val set')
# save train and val loss to pickle
with open('train_val_loss.pkl', 'bw') as f:
    pkl.dump(train_val_loss_list, f)

with open('train_val_acc.pkl', 'bw') as f:
    pkl.dump(train_val_acc_list, f)

# testing
del model
model = best_model
gold_test_list = []
pred_test_list = []
for X, y in test_loader:
    y_pred = model(X)
    gold_test_list.append(np.asarray(y))
    pred_test_list.append(np.asarray(y_pred))

gold_test_list = np.concatenate(gold_test_list)
pred_test_list = np.concatenate(pred_test_list)
pred_test_list[pred_test_list >= 0.5] = 1
pred_test_list[pred_test_list < 0.5] = 0
test_acc = sum(pred_test_list == gold_test_list) / len(pred_test_list)
print(f'Test accuracy is: {test_acc:2f}')
