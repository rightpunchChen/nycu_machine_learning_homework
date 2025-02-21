import numpy as np
import matplotlib.pyplot as plt
from matrixOp import Mat

def vprint(*args):
    test = 0
    if test:
        print(*args)

def print_fl(coe, n):
    f_l = "" # Fitting line
    for i in range(len(coe)):
        f_l += f"{str(abs(coe[i][0]))}"
        if i != n-1: 
            if coe[i+1][0] >= 0: f_l += f'X^{n-i-1} + '
            else: f_l += f'X^{n-i-1} - '
    print(f'Fitting line: {f_l}')

def print_se(err):
    square_err = sum([err[i][0]*err[i][0] for i in range(len(err))])
    print(f'Total error: {square_err}')
    return square_err

def plot_result(n, l, data_x, data_y, LSE_coe, SDM_coe, NM_coe, LSE_err, SDM_err, NM_err):
    x_plot = np.linspace(min(data_x), max(data_x), 500)
    
    # Compute LSE fitting line
    y_lse = np.zeros_like(x_plot)
    for i in range(len(LSE_coe)):
        y_lse += LSE_coe[i][0] * (x_plot ** (n - i - 1))

    # Compute SDM fitting line
    y_sdm = np.zeros_like(x_plot)
    for i in range(len(SDM_coe)):
        y_sdm += SDM_coe[i][0] * (x_plot ** (n - i - 1))

    # Compute NM fitting line
    y_nm = np.zeros_like(x_plot)
    for i in range(len(NM_coe)):
        y_nm += NM_coe[i][0] * (x_plot ** (n - i - 1))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Polynomial Fitting Comparison, Bases: {n}, Lambda: {int(l)}')

    # Plot LSE polynomial fitting line
    axs[0].scatter(data_x, [y[0] for y in data_y], color='red', label='Data Points')
    axs[0].plot(x_plot, y_lse, label='LSE Fitting Line', color='blue')
    axs[0].set_title(f'LSE Fitting Line, error: {LSE_err:.4f}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend()

    # Plot SDM polynomial fitting line
    axs[1].scatter(data_x, [y[0] for y in data_y], color='red', label='Data Points')
    axs[1].plot(x_plot, y_sdm, label='SDM Fitting Line', color='blue')
    axs[1].set_title(f'SDM Fitting Line, error: {SDM_err:.4f}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].legend()

    # Plot Newton method polynomial fitting line
    axs[2].scatter(data_x, [y[0] for y in data_y], color='red', label='Data Points')
    axs[2].plot(x_plot, y_nm, label='Newton method Fitting Line', color='blue')
    axs[2].set_title(f'Newton method Fitting Line, error: {NM_err:.4f}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    with open('testfile.txt', 'r', encoding='utf-8') as file:
        data_x = []
        data_y = []
        for line in file:
            x, y = line.strip().split(',')
            x = float(x)
            y = float(y)
            data_x.append(x)
            data_y.append([y])

    n = int(input("Poly. bases: "))
    l = float(input("Lambda: "))
    
    # Construct phi_n(x) & matrix A, b
    # ========================================================
    A = []
    for i in range(len(data_x)):
        phi = []
        for j in range(n-1, -1, -1):
            # x^n-1, x^n-2, ... , x, 1
            tmp = 1
            for k in range(j):
                tmp = tmp * data_x[i]
            phi.append(tmp)
        A.append(phi)

    A = Mat(A)
    b = Mat(data_y)
    vprint(A)
    vprint(b)
    # ========================================================

    AT = A.T()
    ATA = AT.mul(A)
    ATb = AT.mul(b)
    vprint(ATA)
    lI = [[0.0] * ATA.rows for _ in range(ATA.rows)]
    for i in range(len(lI)): lI[i][i] = 1.0 * l
    lI = Mat(lI)
    vprint(lI)
    ATAlI_inv_ATb = ATA.add(lI).inv().mul(ATb)

    # Regulization LSE
    # ========================================================
    LSE_x_coe = ATAlI_inv_ATb
    LSE_err = A.mul(LSE_x_coe).sub(b)
    vprint(LSE_x_coe)
    vprint(LSE_err)

    print('LSE:')
    print_fl(LSE_x_coe.data, n)
    LSE_err_num = print_se(LSE_err.data)

    # Steepest descent method
    # ========================================================
    epoch = 5000
    learning_rate = 5e-6
    tol = 1e-5
    SDM_x_coe = Mat([[0] for _ in range(A.cols)])  # Start with zero coefficients
    for iter in range(epoch):
        SDM_err = A.mul(SDM_x_coe).sub(b) # Calculate the current error Ax - b

        # gradient = AT.mul(SDM_err)
        gradient = AT.mul(SDM_err).add(SDM_x_coe.sign().scalar_mul(l)) # Compute the gradient

        SDM_x_coe = SDM_x_coe.sub(gradient.scalar_mul(learning_rate)) # Update the coefficients
        vprint(SDM_x_coe)
        if gradient.T().mul(gradient).data[0][0] < tol*tol:
            vprint(f"Converged in {iter}")
            break
    
    SDM_err = A.mul(SDM_x_coe).sub(b)

    print('\nSteepest Descent Method:')
    print_fl(SDM_x_coe.data, n)
    SDM_err_num = print_se(SDM_err.data)

    # Newton method
    # ========================================================
    # NM_x_coe = ATAlI_inv_ATb
    NM_x_coe = ATA.inv().mul(ATb)
    NM_err = A.mul(NM_x_coe).sub(b)
    print('\nNewton\'s Method:')
    print_fl(NM_x_coe.data, n)
    NM_err_num = print_se(NM_err.data)

    # Plot result
    # ========================================================
    plot_result(n, l, data_x, data_y, 
                LSE_x_coe.data, SDM_x_coe.data, NM_x_coe.data,
                LSE_err_num, SDM_err_num, NM_err_num)