# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:24:20 2020

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

import numpy as np
import cv2
import pandas as pd #to handle data as dataframes
import os
import pickle
#############################################################################
#STEP 1######################################################################
#############################################################################
image_dataset = pd.DataFrame()
img_path = "Train_Img/"

for image in os.listdir(img_path):
    print(image)
    df = pd.DataFrame() #temporary data frame to capture information for each loop
    #reset dataframe to blank after each loop
    input_img = cv2.imread(img_path + image)

    #check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #bw img !
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with greyscale and RGB images!")

#Start adding to the dataframe

#Add pixel values to the data frame (first feature)
    pixel_values = img.reshape(-1) #reshape img to a single column
    df["Pixel_Value"] = pixel_values #Pixel value itself as a feature
    df["Image_Name"] = image #Capture image name as we read multiple images
    
#Gabor features (ähnliche gaussian filter oder canny edge)
    #Generate Gabor features
    num = 1 #To count numbers up in order to give Gabor features a Lable in the data frame
    #create gabor "filterbank"
    kernels = []
    for theta in range (2): #define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3): #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4): #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    gabor_label = "Gabor" + str(num)  #Label Gabor columns as Gabor 1, Gabor2,
                    #print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    #kernelsize, sigma changed 2 times, theta changed 2 times, lamda von zero to pi, gamma is 0.05 (high aspect ratio) und 0.5 (medium aspect ratio)))
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel) #filtered img
                    filtered_img = fimg.reshape(-1) #to convert it into a column
                    df[gabor_label] = filtered_img #labels columns as gabor1, gabor2, etc.
                    #print(gabor_label, ": theta=", theta, ": sigma=", sigma, ": lambda=", lamda, ": gamma=", gamma)
                    num += 1 #increment for gabor column label (increase number as it gets to the loop)

##########################       
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
  #  variance_img = nd.generic_filter(img, np.var, size=3) #varianzfunktion sehr langsa, kann auch weggelassen werden
  #  variance_img1 = variance_img.reshape(-1)
  #  df['Variance s3'] = variance_img1  #Add column to original dataframe  
    
    


#labeled_img = cv2.imread("Labeled_Img/Prozess_197_Detail_01_lab.tif")
#labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
#labeled_img1 = labeled_img.reshape(-1)
#df["Label"] = labeled_img1

    image_dataset = image_dataset.append(df)

#############################################################################
#update dataframe for images to include details for each image in the loop
    

#############################################################################
#STEP 2######################################################################
#############################################################################
mask_dataset = pd.DataFrame()

mask_path = "Labeled_Img/"
for mask in os.listdir(mask_path):
    print(mask)
    df2 = pd.DataFrame()
    input_mask = cv2.imread(mask_path + mask)
    
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with greyscale and RGB images!")

    label_values = label.reshape(-1)
    df2["Label_Value"] = label_values
    df2["Mask_Name"] = mask
    
    mask_dataset = mask_dataset.append(df2)
    
#############################################################################
#STEP 3######################################################################
#############################################################################

dataset = pd.concat([image_dataset, mask_dataset], axis=1) #masterdataset
dataset = dataset[dataset.Label_Value !=0] #do not include pixels with value 0 (no label)

X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1) #all features without names/labels (these will be dropped)
Y = dataset["Label_Value"].values #prediction

#Split data into test and train to verify accuracy after fitting the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#############################################################################
#STEP 4######################################################################
#############################################################################

#import ML algorithm and train model
from sklearn.ensemble import RandomForestClassifier #regressor alternative, predict actual value
#regressor -> result = floating point number
model = RandomForestClassifier(n_estimators = 50, random_state=42) #42 random choice; with gpu/tensorflow does not work
#train model on trainig data
### implement alternative witch pytorch?
model.fit(X_train, Y_train)
#ab hier Validierung notwendig

#############################################################################
#STEP 5######################################################################
#############################################################################

from sklearn import metrics
prediction_test = model.predict(X_test)
print("Accuracy =", metrics.accuracy_score(Y_test, prediction_test)) 
#looks at two of Y_test and prints out accuracy

#importances = list(model.feature_importances_) #output list

#Feature Ranking
#feature_list = list(X.columns)
#feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
#print(feature_imp)
#use top 5 ca or top 10 and use onlye these ones

#############################################################################
#STEP 6######################################################################
#############################################################################

#Saving trained models via pickling it

from datetime import date
#current date of the model
now = date.today()
model_name = ("crack_model_"+str(now))

pickle.dump(model, open(model_name, "wb"))    #w - write, b - binary

#load model
#load_model = pickle.load(open(model_name, "rb"))  #r - read, b - binary
#result = load_model.predict(X)

#segmented_img = result.reshape((img.shape))

#from matplotlib import pyplot as plt
#plt.imshow(segmented_img, cmap="jet")
#plt.imsave("segmented_crack.jpg", segmented_img, cmap="jet")

"""
#Feature Ranking
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
#print(feature_imp)
#use top 5 ca or top 10 and use onlye these ones
"""