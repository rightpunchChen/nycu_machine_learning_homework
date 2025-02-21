import os
import time
import numpy as np
from tqdm import tqdm
root = 'ML_HW/HW2/'
path_config = {
    'imagesTr_path': os.path.join(root, 'dataset/train-images.idx3-ubyte_'),
    'labelsTr_path': os.path.join(root, 'dataset/train-labels.idx1-ubyte_'),
    'imagesTs_path': os.path.join(root, 'dataset/t10k-images.idx3-ubyte_'),
    'labelsTs_path': os.path.join(root, 'dataset/t10k-labels.idx1-ubyte_')
}
_time = time.localtime()
_time = time.strftime("%Y_%m_%d_%H_%M_%S", _time)

def dataLoader(state):
    with open(path_config[f'images{state}_path'], 'rb') as f:
        images = {
            'magic': int.from_bytes(f.read(4), 'big'),
            'num': int.from_bytes(f.read(4), 'big'),
            'rows': int.from_bytes(f.read(4), 'big'),
            'cols': int.from_bytes(f.read(4), 'big'),
        }
        # Reshape [6000,28*28]
        images['data'] = np.fromfile(f, dtype=np.uint8).reshape(images['num'], images['rows']*images['cols'])
    with open(path_config[f'labels{state}_path'], 'rb') as f:
        labels = {
            'magic': int.from_bytes(f.read(4), 'big'),
            'num': int.from_bytes(f.read(4), 'big'),
        }
        labels['data'] = np.fromfile(f, dtype=np.uint8)
    return images, labels

def printPostnResult(posterior, label, save_path=''):
    if save_path:
        with open(save_path, 'a') as file:
            file.write("Postirior (in log scale):\n")
            for i in range(10):
                file.write(f"{i}: {posterior[i]}\n")
            file.write(f"Prediction: {np.argmin(posterior)}, Ans: {label}\n")
    else:
        print("Postirior (in log scale):")
        for i in range(10):
            print(f"{i}: {posterior[i]}")
        print(f"Prediction: {np.argmin(posterior)}, Ans: {label}\n")
    

def printImages(img_bins, pixel_num, cols, mode, save_path=''):
    if save_path:
        if mode==0:
            with open(save_path, 'a') as file:
                for l in range(10):
                    file.write(f'{l}:\n')
                    for i in range(pixel_num):
                        x = np.argmax(img_bins[l][i][:])
                        if x>16: file.write('1') 
                        else: file.write('0')
                        if (i+1)%cols == 0: file.write('\n')
                    file.write('\n')
        elif mode==1:
            with open(save_path, 'a') as file:
                for l in range(10):
                    file.write(f'{l}:\n')
                    for i in range(pixel_num):
                        x = img_bins[l][i]
                        if x>=128: file.write('1')
                        else: file.write('0')
                        if (i+1)%cols == 0: file.write('\n')
                    file.write('\n')
    else:
        if mode==0:
            for l in range(10):
                print(f'{l}:')
                for i in range(pixel_num):
                    x = np.argmax(img_bins[l][i][:])
                    if x>16: print(1, end='') 
                    else: print(0, end='')
                    if (i+1)%cols == 0: print()
                print()
        elif mode==1:
            for l in range(10):
                print(f'{l}:')
                for i in range(pixel_num):
                    x = img_bins[l][i]
                    if x>=128: print(1, end='') 
                    else: print(0, end='')
                    if (i+1)%cols == 0: print()
                print()

def discreteMode(bins=8):
    print("discreteMode")
    """
                         P(X|L_k)*P(L_k) 
    argmax P(L_k|X) = ------------------ propotional to {P(X_0|L_k)*...*P(X_n|L_k)}*P(L_k)
                               P(L)
    X_i: i-th pixel value, L_k: label k
    
    take log: log(P(L_k|X)) propotional to {log(P(X_0|L_k))+...+log(P(X_n|L_k))} + log(P(L_k))

    """
    imagesTr, labelsTr = dataLoader('Tr')
    imagesTs, labelsTs = dataLoader('Ts')
    train_num, test_num = imagesTr['num'], imagesTs['num']
    rows, cols = imagesTr['rows'], imagesTr['cols']
    pixel_num = rows*cols

    # Image preprocess 
    # ========================================================
    # Dicretiz image to 0~32
    imagesTr_discrete = imagesTr['data'] // bins
    imagesTs_discrete = imagesTs['data'] // bins 

    # Training 
    # ========================================================
    # Construct bins
    print("Training: ")
    img_bins = np.zeros((10, pixel_num, 256//bins), dtype=np.float64)
    priors = np.zeros(10, dtype=np.float64)
    for n in tqdm(range(train_num)):
        for pixel in range(pixel_num):
                img_bins[labelsTr['data'][n]][pixel][imagesTr_discrete[n][pixel]] +=1
        # Compute prior        
        priors[labelsTr['data'][n]] += 1
    img_bins[img_bins == 0] = 1e-6

    # Testing 
    # ========================================================
    print("Testing: ")
    error_count = 0
    for n in tqdm(range(test_num)):
        # Compute posteriors
        log_posteriors = np.log(priors / train_num).copy()
        for l in range(10):
            # Compute likelihood
            for pixel in range(pixel_num):
                log_posteriors[l] += np.log(img_bins[l][pixel][imagesTs_discrete[n][pixel]] / priors[l])
        # Normalize
        log_posteriors /= sum(log_posteriors)

        # Count error
        if np.argmin(log_posteriors) != labelsTs['data'][n]: error_count += 1

        # Print posterior
        # printPostnResult(log_posteriors, labelsTs['data'][n])
        printPostnResult(log_posteriors, labelsTs['data'][n], os.path.join(root, f'discrete_posterior_{_time}.txt')) # To txt

    # Print image & Error
    # printImages(img_bins, pixel_num, cols, 0)
    printImages(img_bins, pixel_num, cols, 0, os.path.join(root, f'discrete_images_{_time}.txt')) # To txt
    print(f"Error rate: {error_count/test_num}")

def contiMode():
    print("contiMode")
    """
    MLE of Gaussian -> mu_i = mean(pixel_i), sigma_square_i = var(pixel_i)
    1. compute mu & sigma_square for each pixel
    2. likelihood for each pixel of testing data in log: 
        x_i = -1/2log(2*pi*sigma_square_i)+(x_i-mu_i)^2/(2*sigma_square_i)
    3. compute 2. for all label and find min.
    """
    imagesTr, labelsTr = dataLoader('Tr')
    imagesTs, labelsTs = dataLoader('Ts')
    train_num, test_num = imagesTr['num'], imagesTs['num']
    rows, cols = imagesTr['rows'], imagesTr['cols']
    pixel_num = rows*cols

    imagesTr_conti = imagesTr['data']
    imagesTs_conti = imagesTs['data']

    # Training 
    # ========================================================
    print("Training: ")
    mu = np.zeros((10, pixel_num), dtype=np.float64)
    sigma_square = np.zeros((10, pixel_num), dtype=np.float64)
    priors = np.zeros(10, dtype=np.float64)
    for n in tqdm(range(train_num)):
        # Compute mean and var
        priors[labelsTr['data'][n]] += 1
        for pixel in range(pixel_num):
            mu[labelsTr['data'][n]][pixel] += imagesTr_conti[n][pixel]
            sigma_square[labelsTr['data'][n]][pixel] += imagesTr_conti[n][pixel]**2
    for l in range(10):
        count = np.sum(labelsTr['data']==l)
        mu[l][:] /= count
        sigma_square[l][:] /= count
        sigma_square[l][:] -= mu[l][:]**2
        sigma_square[l][:] = np.maximum(sigma_square[l][:], 1e-6)

    # Testing 
    # ========================================================
    print("Testing: ")
    error_count = 0
    for n in tqdm(range(test_num)):
        # Compute posteriors
        log_posteriors = np.log(priors / train_num).copy()
        for l in range(10):
            # Compute likelihood
            for pixel in range(pixel_num):
                # -1/2log(2*pi*sigma_square_i)+(x_i-mu_i)^2/(2*sigma_square_i)
                log_posteriors[l] += ((-0.5)*np.log(2*np.pi*sigma_square[l][pixel]))+(((imagesTs_conti[n][pixel]-mu[l][pixel])**2)/(2*sigma_square[l][pixel]))
        # Normalize
        log_posteriors /= sum(log_posteriors)
        # Count error
        if np.argmin(log_posteriors) != labelsTs['data'][n]: error_count += 1
    # Print posterior
        # printPostnResult(log_posteriors, labelsTs['data'][n])
        printPostnResult(log_posteriors, labelsTs['data'][n], os.path.join(root, f'conti_posterior_{_time}.txt')) # To txt

    # Print image & Error
    # printImages(mu, pixel_num, cols, 1)
    printImages(mu, pixel_num, cols, 1, os.path.join(root, f'conti_images_{_time}.txt')) # To txt
    print(f"Error rate: {error_count/test_num}")


if __name__ == '__main__':
    mode = input('0: discrete mode\n1: continuous mode\n-> ')
    if mode == '0':
        discreteMode()
    elif mode == '1':
        contiMode()

# python ML_HW/HW2/naive_bayes_classifier.py