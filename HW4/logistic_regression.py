import os
import time
import numpy as np
import matplotlib.pyplot as plt

_time = time.localtime()
_time = time.strftime("%Y_%m_%d_%H_%M_%S", _time)
root = 'ML_HW/HW4'

def generator(mean=0, var=1):
    uniform = np.random.uniform(0, 1, 12)
    irwin_hall = np.sum(uniform) - 6
    return float(mean + (irwin_hall* np.sqrt(var)))

def dataLoader(N, mean_x, mean_y, var_x, var_y, label):
    data = []
    label_ = []
    for _ in range(N):
        data.append([generator(mean_x, var_x), generator(mean_y, var_y)])
        label_.append([label])
    return np.array(data), np.array(label_)

def Sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def GD_method(X, y, max_iter=100, tol=1e-2, lr=0.01):
    W = np.zeros((3, 1), dtype=np.float64)
    W_pre = np.zeros_like(W)

    for n in range(max_iter):
        gradiant = X.T @ (y - Sigmoid(X @ W))
        W = W + lr * gradiant
        if(np.linalg.norm(W - W_pre) < tol):
            break
        W_pre = W.copy()
        n+=1
    return W

def Newton_method(X, y, max_iter=1, tol=1e-2, lr=0.01):
    W = np.zeros((3, 1), dtype=np.float64)
    W_pre = np.zeros_like(W)

    for n in range(max_iter):
        # Hessian
        pred = Sigmoid(X @ W)
        D = np.diag((pred * (1 - pred)).flatten())
        H = X.T @ D @ X
        gradiant = X.T @ (y - pred)

        try:
            inv_H = np.linalg.inv(H)
        except:
            W = W + lr * gradiant
        else:
            W = W + (inv_H @ gradiant)

        if(np.linalg.norm(W - W_pre) < tol):
            print(n)
            break

        W_pre = W.copy()
        n+=1
    return W

def result(X, W):
    pred = Sigmoid(X @ W)
    return np.where(pred>0.5, 1, 0)

def metric(gt, pred):
    confusion_mat = np.zeros((2, 2), dtype=np.int8)
    for i in range(len(gt)):
        if pred[i] == 0 and gt[i] == 0 : # TP
            confusion_mat[0][0] += 1
        if pred[i] == 0 and gt[i] == 1 : # FP
            confusion_mat[0][1] += 1
        if pred[i] == 1 and gt[i] == 0 : # FN
            confusion_mat[1][0] += 1
        if pred[i] == 1 and gt[i] == 1 : # TN
            confusion_mat[1][1] += 1
    return confusion_mat

def output(W_gd, confusion_gd, W_Newton, confusion_Newton, save_path='s.txt'):
    with open(save_path, 'a') as file:
        file.write('Gradient descent:\n\nw:\n')
        for i in range(3):
            file.write(f'{W_gd[i].item()}\n')
        file.write('\nConfusion Matrix:\n')
        file.write('\t\t\tPredict cluster 1 Predict cluster 2\n')
        file.write(f'Is cluster 1\t\t{confusion_gd[0][0]}\t\t\t{confusion_gd[0][1]}\n')
        file.write(f'Is cluster 2\t\t{confusion_gd[1][0]}\t\t\t{confusion_gd[1][1]}\n')
        sensitivity = confusion_gd[0][0] / (confusion_gd[0][0] + confusion_gd[0][1])
        specificity = confusion_gd[1][1] / (confusion_gd[1][0] + confusion_gd[1][1])
        file.write(f"\nSensitivity (Successfully predict cluster 1): {sensitivity}\n")
        file.write(f"Specificity (Successfully predict not cluster 2): {specificity}\n\n")

        file.write('Newton method:\n\nw:\n')
        for i in range(3):
            file.write(f'{W_Newton[i].item()}\n')
        file.write('\nConfusion Matrix:\n')
        file.write('\t\t\tPredict cluster 1 Predict cluster 2\n')
        file.write(f'Is cluster 1\t\t{confusion_Newton[0][0]}\t\t\t{confusion_Newton[0][1]}\n')
        file.write(f'Is cluster 2\t\t{confusion_Newton[1][0]}\t\t\t{confusion_Newton[1][1]}\n')
        sensitivity = confusion_Newton[0][0] / (confusion_Newton[0][0] + confusion_Newton[0][1])
        specificity = confusion_Newton[1][1] / (confusion_Newton[1][0] + confusion_Newton[1][1])
        file.write(f"\nSensitivity (Successfully predict cluster 1): {sensitivity}\n")
        file.write(f"Specificity (Successfully predict not cluster 2): {specificity}\n")

if __name__ == '__main__':
    N = int(input('N: '))
    mx1 = float(input('mx1: '))
    my1 = float(input('my1: '))
    mx2 = float(input('mx2: '))
    my2 = float(input('my2: '))
    vx1 = float(input('vx1: '))
    vy1 = float(input('vy1: '))
    vx2 = float(input('vx2: '))
    vy2 = float(input('vy2: '))
    
    data_1, label_1 = dataLoader(N, mx1, my1, vx1, vy1, 0)
    data_2, label_2 = dataLoader(N, mx2, my2, vx2, vy2, 1)
    data = np.concatenate((data_1, data_2))
    label = np.concatenate((label_1, label_2))
    label_f = label.flatten()

    # Design matrix A
    A = np.ones((2*N, 3), dtype=np.float64)
    A[:,:2] = data

    # Gradiant decent
    ###################################################
    W_gd = GD_method(A, label)
    result_gd = result(A, W_gd).flatten()
    confusion_gd = metric(label_f, result_gd)


    # Newton method
    ###################################################
    W_Newton = Newton_method(A, label)
    result_Newton = result(A, W_Newton).flatten()
    confusion_Newton = metric(label_f, result_Newton)


    output(W_gd, confusion_gd, W_Newton, confusion_Newton, save_path=os.path.join(root, f'logistic_regression_{_time}.txt'))

    # Plot
    ##################################################
    fig = plt.figure(figsize=(12, 8))
    # GT
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('Ground Truth')
    ax.scatter(data_1[:,0], data_1[:,1], c='r')
    ax.scatter(data_2[:,0], data_2[:,1], c='b')

    # Gradiant decent
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('Gradiant decent')
    ax.scatter(data[result_gd == 0, 0], data[result_gd == 0, 1], c='r')
    ax.scatter(data[result_gd == 1, 0], data[result_gd == 1, 1], c='b')

    # Newton method
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Newton method')
    ax.scatter(data[result_Newton == 0, 0], data[result_Newton == 0, 1], c='r')
    ax.scatter(data[result_Newton == 1, 0], data[result_Newton == 1, 1], c='b')

    plt.tight_layout()
    plt.savefig(os.path.join(root, f'logistic_regression_{_time}.jpg'))
    plt.show()