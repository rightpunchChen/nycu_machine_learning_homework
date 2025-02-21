import os
import time
import numpy as np
import matplotlib.pyplot as plt
from random_data_genetator import generator_poly

root = 'ML_HW/HW3'
_time = time.localtime()
_time = time.strftime("%Y_%m_%d_%H_%M_%S", _time)

def y_result(x, basis, var, mean, cov_mat):
    # y = mean * w
    _x = np.zeros((len(x), basis))
    y_result = np.zeros((3, len(x)))
    for i in range(len(x)): 
        for j in range(basis):
            _x[i][j] = x[i]**j
    for i in range(len(x)):
        y_result[0][i] = (mean.T @ _x[i][:].T)[0]
        y_result[1][i] = (mean.T @ _x[i][:].T)[0] + _x[i]@cov_mat@_x[i].T + var
        y_result[2][i] = (mean.T @ _x[i][:].T)[0] - (_x[i]@cov_mat@_x[i].T + var)
    return y_result

def write_result(point_x, point_y, mean, var, predict_mean, predict_var):
    with open(os.path.join(root, f'baysian_linear_regression_{_time}.txt'), 'a') as f:
        f.write(f'Add data point ({point_x}, {point_y}):\n\n')
        f.write('Postirior mean:\n\n')
        # for i in range(len(mean)):
        #     f.write(f'{mean[i]}\n')
        for row in mean:
            f.write(f"{' '.join(map(str, row))}\n")
        f.write('\nPostirior variance:\n\n')
        for row in var:
            f.write(f"{' '.join(map(str, row))}\n")
        # for i in range(len(var)):
        #     for j in range(len(var)):
        #         f.write(f'{var[i][j]} ')
        #     f.write(f'\n')
        f.write(f'\nPredictive distribution ~ N({predict_mean}, {predict_var})\n\n')

if __name__ == '__main__':
    precision = int(input('precision: '))
    basis = int(input('basis: '))
    var = float(input('var: '))
    w = []
    for i in range(basis):
        w.append(float(input(f'w[{i}]: ')))
    
    # precision = 1
    # basis = 4
    # var = 1.0
    # w = [1,2,3,4]

    # Initial
    ###################################################
    n = 1
    data_x = []
    data_y = []
    mean = np.zeros((basis, 1), dtype=np.float64)
    previous_mean = np.zeros((basis, 1), dtype=np.float64)
    previous_predict_mean, predict_mean, previous_predict_var, predict_var = 0, 0, 0, 0
    
    cov_mat = np.zeros((basis, basis), dtype=np.float64)
    previous_cov_mat = np.zeros((basis, basis), dtype=np.float64)
    for i in range(basis): previous_cov_mat[i][i] = 1 / precision

    # Compute posterior
    ###################################################
    tolerance = 1e-6
    while(n<2000):
        point_x, point_y = generator_poly(basis, var, w)
        # print(f'Add data point ({x}, {y}):\n')
        data_x.append(point_x); data_y.append(point_y)

        # Compute X
        X = np.zeros((1, basis))
        for i in range(basis): X[0][i] = point_x**i
        
        # Compute w ~ N(C_n((1/v)x^ty + C_{n-1}^-1 m_{n-1}), ((1/v)x^Tx + C_{n-1}^-1)^-1)
        cov_mat = np.linalg.inv((1 / var) * (X.T @ X) + np.linalg.inv(previous_cov_mat))
        mean = cov_mat @ (((1 / var) * X.T * point_y) + (np.linalg.inv(previous_cov_mat) @ previous_mean))

        # Compute y ~ N(m^TX^T, XCX^T+var)
        predict_mean = mean.T @ X.T
        predict_var = (X @ cov_mat @ X.T) + var
        
        if n==10:
            mean_10 = np.zeros_like(mean)
            mean_10 = mean.copy()
            cov_mat_10 = np.zeros_like(cov_mat)
            cov_mat_10 = cov_mat.copy()
        if n==50:
            mean_50 = np.zeros_like(mean)
            mean_50 = mean.copy()
            cov_mat_50 = np.zeros_like(cov_mat)
            cov_mat_50 = cov_mat.copy()

        write_result(point_x, point_y, mean, cov_mat, predict_mean, predict_var)

        if abs(predict_var - previous_predict_var) < tolerance:
            print(f'Converged at step {n}.')
            break
        previous_predict_mean = predict_mean
        previous_predict_var = predict_var
        previous_mean = mean
        previous_cov_mat = cov_mat
        n += 1

    print(f'Final predictive distribution ~ N({predict_mean}, {predict_var}) ')

    # Plot
    ###################################################
    x = np.linspace(-2, 2, 500)
    y_gt = np.zeros_like(x)
    for i in range(basis):
        y_gt += w[i] * (x ** i)
    fig = plt.figure(figsize=(8, 8))
    # GT
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Ground Truth')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-20, 20])
    ax.plot(x, y_gt, 'k')
    ax.plot(x, y_gt + var, 'r')
    ax.plot(x, y_gt - var, 'r')

    # Predict result
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Predict Result')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-20, 20])
    ax.scatter(data_x, data_y)
    predict_result = y_result(x, basis, var, mean, cov_mat)
    ax.plot(x, predict_result[0], 'k')
    ax.plot(x, predict_result[1], 'r')
    ax.plot(x, predict_result[2], 'r')

    if n >= 50:
        # 10 data
        ax = fig.add_subplot(2, 2, 3)
        ax.set_title('After 10 incomes')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-20, 20])
        ax.scatter(data_x[:9], data_y[:9])
        predict_result = y_result(x, basis, var, mean_10, cov_mat_10)
        ax.plot(x, predict_result[0], 'k')
        ax.plot(x, predict_result[1], 'r')
        ax.plot(x, predict_result[2], 'r')

        # 50 data
        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('After 50 incomes')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-20, 20]) 
        ax.scatter(data_x[:49], data_y[:49])
        predict_result = y_result(x, basis, var, mean_50, cov_mat_50)
        ax.plot(x, predict_result[0], 'k')
        ax.plot(x, predict_result[1], 'r')
        ax.plot(x, predict_result[2], 'r')
    plt.tight_layout()
    plt.savefig(os.path.join(root, f'baysian_linear_regression_{_time}.jpg'))
    plt.show()