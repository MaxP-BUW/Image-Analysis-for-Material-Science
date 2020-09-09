# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:58:56 2020

@author: Patzelt
"""

#ridge detection

from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import invert
import cv2

from skimage.filters import meijering, frangi

img = io.imread("Classified image 1.jpg")
#img = rgb2gray(invert(img))
img = rgb2gray(img)

meijering_img = meijering(img)
frangi_img = frangi(img)

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap="gray")
ax1.title.set_text("Input Image")

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(meijering_img, cmap="gray")
ax2.title.set_text("Meijering")

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(frangi_img, cmap="gray")
ax3.title.set_text("Frangi")

plt.show()

"""
io.imsave("meijering.jpg", meijering_img)
io.imsave("sato.jpg", sato_img)
io.imsave("frangi.jpg", frangi_img)
io.imsave("hessian.jpg", hessian_img)
"""

#alle schwarzen Pixel aus input img zählen --> (Summe Px)*Skalierungsfaktor = Fläche

#Mittellinie ermitteln
