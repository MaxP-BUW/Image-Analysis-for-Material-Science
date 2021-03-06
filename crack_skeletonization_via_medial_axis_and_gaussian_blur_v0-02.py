# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:34:08 2021
Last update: Thur Apr 15 13:45:00 2021

@author: Patzelt, Max 
Estimation of crack length and width via medial axis skeletonization

"""
from skimage.morphology import medial_axis
from skimage.filters import threshold_otsu, gaussian
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

img_name = "Classified image 1"
img_type = ".tif"
###READ
#img = imread(img_name+img_type)#, as_gray = True, plugin=None)
img = tiff.imread(img_name+img_type)
img = np.array(img.astype(np.int))
###GAUSSIAN
gaussian_img = gaussian(img, sigma=7)
###SEGMENTATION
thresh = threshold_otsu(gaussian_img)
binary = np.array(gaussian_img > thresh, dtype=int)
###INVERT if needed
#image0 = 1-binary
image0 = binary
###SKELETONIZATION
img_skeletonized, distance = medial_axis(image0, return_distance=True)
img_skeletonized = np.array(img_skeletonized.astype(np.int))
dist_on_skel = distance * img_skeletonized
width = dist_on_skel[dist_on_skel !=0]
###SAVE SKELETON
tiff.imsave(img_name+"_blurred-skeletonized"+img_type, img_skeletonized, 
            shape=None, dtype=int)
###LENGTH+WIDTH
length = np.count_nonzero(img_skeletonized)
mean_width = np.mean(width)
max_width = np.max(width)
min_width = np.min(width)
###OUTPUT
print("Summarized Cracklength = "+str(length)+" Px")
print("Mean crack width = "+str(mean_width)+" Px")
print("Max crack width = "+str(max_width)+" Px")
print("Min crack width = "+str(min_width)+" Px")

#pruning oder fibre analysis will be added next
#find connected items -> all_labels = measure.label(img_skeletonized) 

###PLOT
fs_plot = 70
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(32, 32), sharex=True, 
                         sharey=True)
ax = axes.ravel()
ax[0].imshow(img, cmap="gray_r")
ax[0].axis('off')
ax[0].set_title("Segmented Image", fontsize=fs_plot)
ax[1].imshow(gaussian_img, cmap="gray_r")
ax[1].axis('off')
ax[1].set_title("Blurred Image", fontsize=fs_plot)
ax[2].imshow(binary, cmap="gray_r")
ax[2].axis('off')
ax[2].set_title("Threshold Otsu", fontsize=fs_plot)
ax[3].imshow(img_skeletonized, cmap="gist_heat_r")
#alternative Farbcodes: RdGy, Dark2, nipy_spectral, ocean_r
ax[3].contour(img, [0.1], colors="gray")
ax[3].axis('off')
ax[3].set_title("Skeletonized Image", fontsize=fs_plot)
fig.tight_layout()
plt.show()
