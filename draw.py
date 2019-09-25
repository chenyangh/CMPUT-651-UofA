import pickle as pkl
import matplotlib.pyplot as plt


def draw_two_curves_with_postfix(post_fix=''):
    with open(f'train_val_loss_{post_fix}.pkl', 'br') as f:
        train_val_loss_list = pkl.load(f)

    train_loss_list = [x[0] for x in train_val_loss_list]
    val_loss_list = [x[1] for x in train_val_loss_list]

    x = [x + 1 for x in list(range(len(train_val_loss_list)))]

    plt.plot(x, train_loss_list, 'r', label='Train')
    plt.plot(x, val_loss_list, 'b', label='Val')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'img/train_curve_loss_{post_fix}.png')

    plt.figure()
    with open(f'train_val_acc_{post_fix}.pkl', 'br') as f:
        train_val_acc_list = pkl.load(f)

    train_acc_list = [x[0] for x in train_val_acc_list]
    val_acc_list = [x[1] for x in train_val_acc_list]

    x = [x + 1 for x in list(range(len(train_val_loss_list)))]

    plt.plot(x, train_acc_list, 'r', label='Train')
    plt.plot(x, val_acc_list, 'b', label='Val')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'img/train_curve_acc_{post_fix}.png')


if __name__ == '__main__':
    draw_two_curves_with_postfix()
    draw_two_curves_with_postfix('nn')
