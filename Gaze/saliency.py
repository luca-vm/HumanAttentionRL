import cv2
import numpy as np
import tarfile
import io
import os
import re
import matplotlib.pyplot as plt
from Change_itti_Saliency import Itti_Saliency_map


# Function to extract the numeric part of the filename for sorting
def extract_numeric_part(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

# Function to extract images from tar.bz2 file and get file names
def extract_images_from_tar(tar_path):
    file_images = []
    with tarfile.open(tar_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
                # Decode the image in RGB format
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                file_images.append((member.name, img))
    # Sort images numerically by the number at the end of the filename
    file_images.sort(key=lambda x: extract_numeric_part(x[0]))
    sorted_file_names, sorted_images = zip(*file_images)
    return list(sorted_images), list(sorted_file_names)


# Function to compute optical flow using Farnebäck method
def compute_saliency(images):
    sal_images = []
    for i in range(len(images)):   
        sal = Itti_Saliency_map(images[i])
        sal_images.append(sal)
    return sal_images

# Function to save optical flow images into a tar.bz2 file with the same names as the original images
def save_saliency_to_tar(sal_images, file_names, output_tar_path):
    with tarfile.open(output_tar_path, 'w:bz2') as tar:
        for img, name in zip(sal_images, file_names):
            img_encoded = cv2.imencode('.png', img)[1].tobytes()
            img_info = tarfile.TarInfo(name=name.replace('.png', '.png'))
            img_info.size = len(img_encoded)
            tar.addfile(img_info, io.BytesIO(img_encoded))

# Function to plot images
def plot_images(original_images, sal_images):
    plt.figure(figsize=(12, 8))
    for i in range(min(3, len(original_images))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title(f'Saliency Image {i+1}')
        plt.axis('off')
        
        plt.subplot(2, 3, i + 4)
        plt.imshow(sal_images[i])
        plt.title(f'Saliency Flow Image {i+1}')
        plt.axis('off')
    plt.show()

# Main function
def main():
    tar_path = './ms_pacman/52_RZ_2394668_Aug-10-14-52-42.tar.bz2'
    output_tar_path = './ms_pacman/52_RZ_2394668_Aug-10-14-52-42_sal.tar.bz2'
    # tar_path = './ms_pacman/test.tar.bz2'
    # output_tar_path = './ms_pacman/test_sal.tar.bz2'
    
    # Extract images and file names
    images, file_names = extract_images_from_tar(tar_path)
    
    # Compute optical flow
    sal_images = compute_saliency(images)
    
    # Save optical flow images to tar.bz2 with the original names
    save_saliency_to_tar(sal_images, file_names, output_tar_path)
    
    # Plot the first 3 images and their corresponding optical flow
    plot_images(images[:3], sal_images[:3])

if __name__ == "__main__":
    main()
