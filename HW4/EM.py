import os
import time
import numpy as np
from numba import jit
from numba.core.errors import NumbaWarning
import warnings, time
warnings.simplefilter('ignore', category=NumbaWarning)

_time = time.localtime()
_time = time.strftime("%Y_%m_%d_%H_%M_%S", _time)

path_config = {
    'imagesTr_path': 'ML_HW/HW2/dataset/train-images.idx3-ubyte_',
    'labelsTr_path': 'ML_HW/HW2/dataset/train-labels.idx1-ubyte_',
    'save_path': f'ML_HW/HW4/EM_{_time}.txt'
}

def dataLoader():
    with open(path_config[f'imagesTr_path'], 'rb') as f:
        images = {
            'magic': int.from_bytes(f.read(4), 'big'),
            'num': int.from_bytes(f.read(4), 'big'),
            'rows': int.from_bytes(f.read(4), 'big'),
            'cols': int.from_bytes(f.read(4), 'big'),
        }
        # Reshape [6000,28*28]
        images['data'] = np.fromfile(f, dtype=np.uint8).reshape(images['num'], images['rows']*images['cols'])
        images['data'] = np.where(images['data'] > 127, 1, 0)

    with open(path_config[f'labelsTr_path'], 'rb') as f:
        labels = {
            'magic': int.from_bytes(f.read(4), 'big'),
            'num': int.from_bytes(f.read(4), 'big'),
        }
        labels['data'] = np.fromfile(f, dtype=np.uint8)
    return images, labels

def printImages(P, pixel_num, cols, digit_map, labeled=False, iter='', diff='', save_path=path_config['save_path']):
    with open(save_path, 'a') as file:
        for l in range(10):
            if labeled:
                file.write(f'Labeled ')
            file.write(f'Class {l}:\n')
            digit_class = digit_map[l]
            for i in range(pixel_num):
                if P[digit_class][i] > 0.5: file.write('1')
                else: file.write('0')
                if (i+1)%cols == 0: file.write('\n')
            file.write('\n')
        if iter and diff:
            file.write(f'No. of Iteration: {iter}, Difference: {diff}\n\n')

@jit           
def E_step(img, train_num, pixel_num, L, P):
    """
    Calculate the responsibility matrix W for each digit
    W[i][digit] = l[digit] * ㄇ_k(p[digit])^img[i][k] * ㄇ_k(1-p[digit])^(1-img[i][k]) / W
    """
    W = np.zeros((train_num, 10), dtype=np.float64) # Responsibility of the digit

    for i in range(train_num):
        for j in range(10):
            W[i][j] = L[j]
            for k in range(pixel_num):
                W[i][j] *= (P[j][k] ** img[i][k]) * ((1 - P[j][k]) ** (1 - img[i][k]))
        W[i] /= sum(W[i])

    # for i in range(10):
    #     W[:,i] = L[i] * (P[i] ** img).prod(axis=1) * ((1 - P[i]) ** (1 - img)).prod(axis=1)
    # W /= W.sum(axis=1, keepdims=True)
    return W


def M_step(img, train_num, pixel_num, L, P, W):
    """
    Update the parameters L and P based on the current responsibilities W
    """
    sum_W = W.sum(axis=0)
    sum_W = np.where(sum_W == 0.0, 1e-5, sum_W)
    L = sum_W / train_num

    # for i in range(10):
    #     for j in range(pixel_num):
    #         P[i][j] = img[:, j].T @ W[:, i] / sum_W[i]
    
    P = (W.T @ img) / sum_W[:, None]
    P = np.where(P == 0.0, 1e-5, P)
    return L, P

def give_label(W, labels, train_num):
    labeled_clusters = np.zeros(10)
    tabel = np.zeros((10, 10)) # tabel[label][cluster]
    for i in range(train_num):
        # Find the predicted class for each data
        cluster = np.argmax(W[i])
        tabel[labels[i]][cluster] += 1
    for _ in range(10):
        label, _cluster = np.unravel_index(np.argmax(tabel), tabel.shape)
        # Assign the predicted class to the true label in the mapping
        labeled_clusters[label] = _cluster
        tabel[:, _cluster] = 0
        tabel[label, :] = 0
    return labeled_clusters.astype(np.int8) # digit_map[i] = a certain class e.g digit_map[1]=2 : label 1 is class 2

def metric(labels, W, digit_map, train_num):
    confusion_mat = np.zeros((10, 2, 2))
    for i in range(train_num):
        cluster = np.argmax(W[i])
        pred = np.where(digit_map == cluster)[0].item()
        gt = labels[i]
        for digit in range(10):
            if pred == digit and gt == digit : # TP
                confusion_mat[digit][0][0] += 1
            if pred == digit and gt != digit : # FP
                confusion_mat[digit][0][1] += 1
            if pred != digit and gt == digit : # FN
                confusion_mat[digit][1][0] += 1
            if pred != digit and gt != digit : # TN
                confusion_mat[digit][1][1] += 1
    return confusion_mat

def result(W, labels, train_num, digit_map, iter, save_path=path_config['save_path']):
    err = train_num
    cm = metric(labels, W, digit_map, train_num)
    with open(save_path, 'a') as file:
        for i in range(10):
            file.write("--------------------------------------------------------\n")
            file.write(f"Confusion Matrix {i}:\n")
            file.write(f"\t\tPredict {i}\t\tPredict not {i}\n")
            file.write(f"Is {i}\t{cm[i][0][0]}\t\t\t{cm[i][0][1]}\n")
            file.write(f"Isn't {i}\t{cm[i][1][0]}\t\t\t{cm[i][1][1]}\n")
            sensitivity = cm[i][0][0] / (cm[i][0][0] + cm[i][0][1])
            specificity = cm[i][1][1] / (cm[i][1][0] + cm[i][1][1])
            file.write(f"\nSensitivity (Successfully predict number {i}): {sensitivity}\n")
            file.write(f"Specificity (Successfully predict not number {i}): {specificity}\n")
            err -= cm[i][0][0]
        file.write("--------------------------------------------------------\n")
        file.write(f"Total iteration to converge: {iter}\n")
        file.write(f"Total error rate: {err/train_num}\n")

if __name__ == '__main__':
    images, labels = dataLoader()
    img, train_num, rows, cols = images['data'], images['num'], images['rows'], images['cols']
    pixel_num = rows * cols

    # Initial parameter
    L = 0.1 * np.ones(10, dtype=np.float64) # The probability of the digit appearing P(digit)
    P = 0.2 + 0.6 * np.random.rand(10, pixel_num).astype(np.float64) # The probability of each pixel P(pixel i value|digit)
    P_pre = np.zeros_like(P)

    n = 1
    tolerance = 0.3
    while n < 20:
        print(f'No. of Iteration: {n}')

        W = E_step(img, train_num, pixel_num, L, P)
        L, P = M_step(img, train_num, pixel_num, L, P, W)

        diff = np.linalg.norm(P - P_pre)
        printImages(P, pixel_num, cols, digit_map=np.array([i for i in range(10)]),iter=str(n), diff=str(diff))

        if diff < tolerance: break
        P_pre = P.copy()
        n+=1
    W = E_step(img, train_num, pixel_num, L, P)
    labeled_clusters = give_label(W, labels['data'], train_num)
    printImages(P, pixel_num, cols, digit_map=labeled_clusters, labeled=True)
    result(W, labels['data'], train_num, labeled_clusters, str(n))