import os
import time
from random_data_genetator import generator

root = 'ML_HW/HW3'
_time = time.localtime()
_time = time.strftime("%Y_%m_%d_%H_%M_%S", _time)
if __name__ == '__main__':
    m = float(input('mean: '))
    s = float(input('var: '))

    with open(os.path.join(root, f'sequential_estimator_{_time}.txt'), 'a') as f:
        f.write(f'Data point source function: N({m}, {s})\n\n')
        # print(f'Data point source function: N({m}, {s})\n')
        tolerance = 1e-6
        previous_mean, mean = 0, 0
        previous_var, var = 0, 0
        n = 1
        while(n<5000):
            new_point = generator(m, s)
            f.write(f'Add data point: {new_point}\n')
            # print(f'Add data point: {new_point}')

            """
            Welford's online algorithm
            mean_n = mean_n-1 + (x - mean_n-1) / n
            var_n = (var_n-1 + (x - mean_n-1)(x - mean_n)) / n
            reference: https://changyaochen.github.io/welford/
            """
            
            tmp = new_point - mean
            mean += tmp / n
            var += (tmp * (new_point - mean))

            f.write(f'Mean = {mean}, Variance = {var/n}\n')
            # print(f'Mean = {mean}, Variance = {var/n}')
            if abs(mean - previous_mean) < tolerance and abs(var - previous_var) < tolerance:
                f.write(f'Converged at step {n}.')
                # print(f'Converged at step {n}.')
                break

            previous_mean = mean
            previous_var = var
            n += 1