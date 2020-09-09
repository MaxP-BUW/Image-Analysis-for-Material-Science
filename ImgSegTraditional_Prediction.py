# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:29:57 2020

@author: Patzelt

_____________________________________________________

Image Segmentation using traditional machine learning
_____________________________________________________

Feauture based segmentation using Random Forest
Using multiple training images

STEP 1: Read training images and extract features
STEP 2: Read labeled images (masks) and create another dataframe
STEP 3: Get data ready for random forest (or other classifier)
STEP 4: Define the classifier and fit the model using traning data
STEP 5: check accuracy of the model
STEP 6: save model for future use
STEP 7: make prediction of new images   

"""
#############################################################################
#STEP 7######################################################################
#############################################################################
import numpy as np
import cv2
import pandas as pd

def feature_extraction(img):
    df = pd.DataFrame()
    
    img2 = img.reshape(-1)
    df["Original Image"] = img2
    #Generate Gabor Features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3): #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4): #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    gabor_label = "Gabor" + str(num)  #Label Gabor columns as Gabor 1, Gabor2,
                    #print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel) #filtered img
                    filtered_img = fimg.reshape(-1) #to convert it into a column
                    df[gabor_label] = filtered_img #labels columns as gabor1, gabor2, etc.
                    num += 1 #increment for gabor column label (increase number as it gets to the loop)

##########################       very import: same features as input images!
#print(df.head)  
#Genarate other features and add them to the data frame   

#Canny edge
    edges = cv2.Canny(img, 100, 200) #input, min, max
    edges1 = edges.reshape(-1)
    df["Canny Edge"] = edges1

    from skimage.filters import roberts, sobel, scharr, prewitt

#roberts edge                
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df["Roberts"] = edge_roberts1

#sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df["Sobel"] = edge_sobel1

#scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df["Scharr"] = edge_scharr1
  
#prewitt              
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df["Prewitt"] = edge_prewitt1 

#GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

#GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

#MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

#VARIANCE with size=3
   # variance_img = nd.generic_filter(img, np.var, size=3) #varianzfunktion sehr langsa, kann auch weggelassen werden
   # variance_img1 = variance_img.reshape(-1)
   # df['Variance s3'] = variance_img1  #Add column to original dataframe  
    
    return df #pandas dataframe returned from function


import pickle
from matplotlib import pyplot as plt
"""___________________________VOR DEM START ANPASSEN:______________________"""
filename = "crack_model_2020-09-08" #ggf. Nutzerabfrage welches Modell gewählt werden soll
loaded_model = pickle.load(open(filename, "rb"))
path = "Test_Img/"
import os
for image in os.listdir(path):
    print(image)
    img1 = cv2.imread(path+image)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    
    plt.imsave("Segmented_Img/"+ image, segmented, cmap = "Greys")