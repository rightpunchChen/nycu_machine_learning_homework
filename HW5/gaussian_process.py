import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.optimize import minimize

def dataLoader(file):
    X = []
    Y = []
    with open(file, "r") as f:
        for data_point in f:
            X.append([float(data_point.split(" ")[0])])
            Y.append([float(data_point.split(" ")[1])])
    return np.array(X), np.array(Y)

def rqKernel(X, Y, para):
    # ||X-Y||^2
    sq_dist = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)

    # sigma^2 * (1 + (||X-Y||^2 / (2 * alpha * l^2)))^(-alpha)
    kernel = (para[0]**2) * (1 + (sq_dist / (2*para[1]*(para[2]**2))))**(-para[1])
    return kernel

def likelihoodFunc(kernel_para, X, Y, beta, kernel):
    # - ln P(Y|para) = 1/2(ln(|C|) + Y'(C^-1)Y + N*ln(2pi))
    K_xx = kernel(X, X, kernel_para.flatten())
    C = K_xx + (1/beta * np.identity(X.shape[0], dtype=np.float64))
    ln_p = 0.5 *((np.log(det(C))) + (Y.T @ inv(C) @ Y) + len(X) * np.log(2 * np.pi))
    return ln_p

def gaussian_process(X, Y, X_pred, kernel, beta, kernel_para):
    """
    mu' = K(X',X) * (K(X,X) + varI)^-1 * Y
    C'  = (K(X',X') + varI) - K(X',X) * (K(X,X) + varI)^-1 * K(X,X')
    """
    # C = (K(X,X) + varI)^-1
    K_xx = kernel(X, X, kernel_para)
    C = K_xx + (1/beta * np.identity(X.shape[0], dtype=np.float64))

    K_xxp = kernel(X, X_pred, kernel_para)
    K_xpxp = kernel(X_pred, X_pred, kernel_para) + (1/beta * np.identity(X_pred.shape[0], dtype=np.float64))

    M_pred = K_xxp.T @ (inv(C)) @ Y
    Var_pred = K_xpxp - K_xxp.T @ (inv(C)) @ K_xxp
    return M_pred, Var_pred

def plot_result(X, Y, X_pred, M_pred, Var_pred, kernel_para):
    title = f"Gaussian process using RQ kernel\nw/ sigma: {int(kernel_para[0])}, alpha: {int(kernel_para[1])}, l: {int(kernel_para[2])}"
    # 95% confidence interval (1.96 * sigma)
    upper_bound = M_pred.flatten() + 1.96 * np.sqrt(np.diag(Var_pred))
    lower_bound = M_pred.flatten() - 1.96 * np.sqrt(np.diag(Var_pred))

    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(X, Y, color='r', label="Data points") # Data points
    plt.plot(X_pred.flatten(), M_pred.flatten(), label="Mean prediction")  # Mean of function
    # 95% confidence interval of function
    plt.fill_between(X_pred.flatten(), upper_bound, lower_bound, alpha=0.3, label="95% Confidence Interval")
    plt.title(title)
    plt.xlim(-60, 60)
    plt.legend() 
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_path = "data/input.data"
    X, Y = dataLoader(data_path)
    X_pred = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    beta = 5

    """
    Task 1
    """
    kernel_para = np.random.randint(1, 10, size=3) # Sigma, Alpha, Length
    M_pred, Var_pred = gaussian_process(X, Y, X_pred, rqKernel, beta, kernel_para)
    plot_result(X, Y, X_pred, M_pred, Var_pred, kernel_para)

    """
    Task 2
    """
    minimized_para = minimize(likelihoodFunc, kernel_para, args=(X, Y, beta, rqKernel))
    M_pred, Var_pred = gaussian_process(X, Y, X_pred, rqKernel, beta, minimized_para.x)
    plot_result(X, Y, X_pred, M_pred, Var_pred, minimized_para.x)