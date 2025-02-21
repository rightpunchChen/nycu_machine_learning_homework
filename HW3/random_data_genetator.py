import numpy as np
import matplotlib.pyplot as plt

def generator(mean=0, var=1):
    uniform = np.random.uniform(0, 1, 12)
    irwin_hall = np.sum(uniform) - 6
    return mean + (irwin_hall* np.sqrt(var))

def generator_poly(n, var, w):
    x = np.random.uniform(-1, 1)
    y = 0.0
    for i in range(n):
        y += w[i] * (x**i)
    y += generator(0, var)
    return float(x), float(y)

if __name__ == '__main__':
    case = input('case(a or b or p): ')
    if case == 'a':
        m = float(input('mean: '))
        s = float(input('var: '))
        print(generator(m, s))
    elif case == 'b':
        n = int(input('basis: '))
        s = float(input('var: '))
        w = []
        for i in range(n):
            w.append(float(input(f'w[{i}]: ')))
        print(generator_poly(n, s, w))
    elif case == 'p':
        mean, std_dev = 0, 1
        samples = [generator(mean, std_dev) for _ in range(10000)]
        plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
        x = np.linspace(-6, 6, 1000)
        plt.plot(x, 1/(std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean)/std_dev)**2), color='r')
        plt.title('Irwin-Hall Distribution vs Standard Normal Distribution')
        plt.show()

        