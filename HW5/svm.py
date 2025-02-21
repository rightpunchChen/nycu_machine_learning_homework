import csv
import numpy as np
from libsvm.svmutil import *

config = {
    'imagesTr': "data/X_train.csv",
    'labelsTr': "data/Y_train.csv",
    'imagesTs': "data/X_test.csv",
    'labelsTs': "data/Y_test.csv"
}

"""
linear: u'*v, polynomial: (gamma*u'*v + coef0)^degree, RBF: exp(-gamma*|u-v|^2)
-d degree : degree in kernel function (default 3)
-g gamma : gamma in kernel function (default 1/num_features)
-r coef0 : coef0 in kernel function (default 0)
-c cost : parameter C of C-SVC (default 1)
-v n : n-fold cross validation
"""
kernel_para = {
    'degree' : [2, 3, 4],
    'gamma' : [1/784, 1e-2, 1e-1, 1],
    'coef0' : [-1, 0, 1, 2],
    'cost' : [1e-2, 1e-1, 1, 10, 100],
    'n' : 3
}

def dataLoader():
    with open(config["imagesTr"], 'r') as f:
        imagesTr = np.array(list(csv.reader(f))).astype(np.float64)
    with open(config["imagesTs"], 'r') as f:
        imagesTs = np.array(list(csv.reader(f))).astype(np.float64)
    with open(config["labelsTr"], 'r') as f:
        labelsTr = np.array(list(csv.reader(f))).astype(np.float64).flatten()
    with open(config["labelsTs"], 'r') as f:
        labelsTs = np.array(list(csv.reader(f))).astype(np.float64).flatten()
    return imagesTr, labelsTr, imagesTs, labelsTs

def gridSearch(images, labels, kernel):
    max_acc = 0
    final_para = ''
    n = kernel_para['n']
    if kernel == 0: # Linear
        for c in kernel_para['cost']:
            para = f'-t 0 -c {c}'
            print(f'para: {para}')
            acc = svm_train(labels, images, para + f' -v {n} -q')
            if acc > max_acc:
                max_acc = acc
                final_para = para
    if kernel == 1: # Polynomial
        for c in kernel_para['cost']:
            for r in kernel_para['coef0']:
                for g in kernel_para['gamma']:
                    for d in kernel_para['degree']:
                        para = f'-t 1 -c {c} -r {r} -g {g} -d {d}'
                        print(f'para: {para}')
                        acc = svm_train(labels, images, para + f' -v {n} -q')
                        if acc > max_acc:
                            max_acc = acc
                            final_para = para
    if kernel == 2: # RBF
        for c in kernel_para['cost']:
            for g in kernel_para['gamma']:
                para = f'-t 2 -c {c} -g {g}'
                print(f'para: {para}')
                acc = svm_train(labels, images, para + f' -v {n} -q')
                if acc > max_acc:
                    max_acc = acc
                    final_para = para
    if kernel == 4: # Linear+RBF
        for c in kernel_para['cost']:
            for g in kernel_para['gamma']:
                para = f'-t 4 -c {c} -g {g}'
                print(f'para: {para}')
                acc = svm_train(labels, images, para + f' -v {n} -q')
                if acc > max_acc:
                    max_acc = acc
                    final_para = para
    return max_acc, final_para

def linearRBF_kernel(u, v, gamma=1/784):
    linear = u @ v.T
    sq_dist = np.sum(u**2, axis=1)[:, None] + np.sum(v**2, axis=1)[None, :] - 2 * (u @ v.T)
    RBF = np.exp(-gamma * sq_dist)
    linearRBF = linear + RBF
    idx = np.arange(1, linearRBF.shape[0] + 1).reshape(-1, 1)
    linearRBF = np.hstack((idx, linearRBF))
    return linearRBF

if __name__ == '__main__':
    imagesTr, labelsTr, imagesTs, labelsTs = dataLoader()
    task = input('Task (1, 2, 3): ')

    """
    Task 1 
    """
    if task == "1":
        kernel = input('Kernel (0 -- Linear, 1 -- Poly, 2 -- RBF): ')
        model = svm_train(labelsTr, imagesTr, f'-t {kernel}')
        pred_labels, acc, pred_values = svm_predict(labelsTs, imagesTs, model)

    """
    Task 2
    """
    if task == "2":
        kernel = int(input('Kernel (0 -- Linear, 1 -- Poly, 2 -- RBF): '))
        acc, para = gridSearch(imagesTr, labelsTr, kernel)
        print(f"Max cross val acc {acc} w/ para {para}")
        model = svm_train(labelsTr, imagesTr, f'{para} -q')
        pred_labels, acc, pred_values = svm_predict(labelsTs, imagesTs, model)

    """
    Task 3
    """
    if task == "3":
        linearRBF_Tr = linearRBF_kernel(imagesTr, imagesTr)
        linearRBF_Ts = linearRBF_kernel(imagesTs, imagesTr)

        acc, para = gridSearch(linearRBF_Tr, labelsTr, 4)
        print(f"Max cross val acc {acc} w/ para {para}")
        model = svm_train(labelsTr, linearRBF_Tr, f'{para} -q')
        pred_labels, acc, pred_values = svm_predict(labelsTs, linearRBF_Ts, model)

        # model = svm_train(labelsTr, linearRBF_Tr, f'-t 4')
        # pred_labels, acc, pred_values = svm_predict(labelsTs, linearRBF_Ts, model)