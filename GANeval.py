# Import the necessary libraries
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from statistics import mean

# Create a function to read the images from the folders
def load_images(folder, flag):
    images = []
    for filename in os.listdir(folder):
        # Read the image from the folder
        img = cv2.imread(os.path.join(folder, filename), flag)
        if img is not None:
            # Add the new image at the end of the list
            images.append(img)
    return images

# Calculate the Peak Signal-to-Noise Ratio
def PSNR(original, reconstructed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(original)
    im2 = Image.fromarray(reconstructed)
    return peak_signal_noise_ratio(np.array(im), np.array(im2))

### Calculate the SSIM ###
def SSIM(original, reconstructed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(original)
    im2 = Image.fromarray(reconstructed)
    return structural_similarity(np.array(im), np.array(im2), multichannel=True)

def main():
    # Read each category of images
    damaged = load_images(r'C:\Licenta\GAN_IMAGES\results_GAN_bs32\32_batch_size_100_epochs', 1)
    damaged1 = load_images(r'C:\Licenta\GAN_IMAGES\results_GAN_bs32\16_batch_size_100_epochs', 1)
    original = load_images(r'C:\Licenta\test_batch\original', 1)

    n = len(damaged)
    psnr_list = []
    ssim_list =[]

    psnr_list1 = []
    ssim_list1 = []

    for i in range(n):
        psnr = round(PSNR(original[i], damaged[i]), 2)
        psnr_list.append(psnr)

        psnr1 = round(PSNR(original[i], damaged1[i]), 2)
        psnr_list1.append(psnr1)

        ssim = round(SSIM(original[i], damaged[i]), 2)
        ssim_list.append(ssim)

        ssim1= round(SSIM(original[i], damaged1[i]), 2)
        ssim_list1.append(ssim1)


    plt.subplot(1, 2, 1)
    plt.hist([psnr_list, psnr_list1], alpha=0.5, label=["32 batch size", "16 batch size"])
    plt.title('PSNR Histogram for GAN', fontsize=20), plt.xlabel('PSNR value [dB]', fontsize=16), plt.ylabel('No. of images', fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist([ssim_list, ssim_list1], alpha=0.5, label=["32 batch size", "16 batch size"])
    plt.title('SSIM Histogram for GAN', fontsize=20), plt.xlabel('SSIM value', fontsize=16), plt.ylabel('No. of images', fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


    return

if __name__ == '__main__':
    main()