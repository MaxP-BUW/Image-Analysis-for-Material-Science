# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:47:53 2020

@author: Patzelt
"""
import cv2

filename = "Classified image 2_grey"
filetype = ".tif"

img = cv2.imread(filename+filetype, cv2.IMREAD_GRAYSCALE)
#cv2.imshow("Input", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

_ , img_binary = cv2.threshold(img, 85,255, cv2.THRESH_BINARY) #"_" because threshold value is given back
#cv2.imshow("Binary", img_binary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#_____________________________________________________________________________
#SKELETONIZATION
from skimage.morphology import medial_axis, skeletonize, thin
import matplotlib.pyplot as plt
from skimage.util import invert, img_as_float
from scipy import misc, ndimage

img_invert = invert(img_binary) #invert

img_invert = img_as_float(img_invert) #255 -> 1 setzen ---> int8 (0:255) to float (0:1)
img_skeleton = skeletonize(img_invert > 0)  #skeletonize
img_skeleton = img_as_float(img_skeleton)

##misc.imsave(filename+" skeletonized"+filetype, invert(img_skeleton))

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(img_invert, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

##misc.imsave(filename+" Medial Axis"+filetype, invert(img_as_float(skel)))
##misc.imsave(filename+" Medial Axis2"+filetype, invert(img_as_float(dist_on_skel)))

#thinned = thin(img)
#skeleton_lee = skeletonize(img_invert)

#_____________________________________________________________________________
#PRUNING




#_____________________________________________________________________________
#ANALYSE SKELETON



#_____________________________________________________________________________
#CRACK AREA DIVIDED BY LONGEST PATH LENGTH



#_____________________________________________________________________________
#print('\007') #sound - finished