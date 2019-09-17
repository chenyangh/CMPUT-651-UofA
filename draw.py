import pickle as pkl
import matplotlib.pyplot as plt

with open('train_val_loss.pkl', 'br') as f:
    train_val_loss_list = pkl.load(f)

train_loss_list = [x[0] for x in train_val_loss_list]
val_loss_list = [x[1] for x in train_val_loss_list]

x = [x + 1 for x in list(range(len(train_val_loss_list)))]

plt.plot(x, train_loss_list, 'r', label='Train')
plt.plot(x, val_loss_list, 'b', label='Val')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('img/train_curve_loss.png')

plt.figure()
with open('train_val_acc.pkl', 'br') as f:
    train_val_acc_list = pkl.load(f)

train_acc_list = [x[0] for x in train_val_acc_list]
val_acc_list = [x[1] for x in train_val_acc_list]

x = [x + 1 for x in list(range(len(train_val_loss_list)))]

plt.plot(x, train_acc_list, 'r', label='Train')
plt.plot(x, val_acc_list, 'b', label='Val')
plt.ylabel('Loss')
plt.xlabel('Accuracy')
plt.legend()
plt.savefig('img/train_curve_acc.png')
