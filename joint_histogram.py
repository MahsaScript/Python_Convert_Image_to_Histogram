# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 05:12:57 2021

@author: mahsa
"""

from PIL import Image

img = Image.open('1.jpg')

img2 = Image.open('2.jpg')

# images of size n x p
n=220 #width
p=180 #height
Nimg = img.resize((n,p))   # image resizing 220*180

Nimg2 = img2.resize((n,p)) # image resizing
Nimg.save('11.jpg')
Nimg2.save('22.jpg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.figure()
plt.imshow(Nimg) 
plt.show()  # display image

plt.figure()
plt.imshow(Nimg2) 
plt.show()  # display image

import cv2
image = cv2.imread('11.jpg')  # image reading first image 
image2 = cv2.imread('22.jpg') # image reading second image

gray_image  = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY) # Converting to gray first image
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # Converting to gray second image

histogram1 = cv2.calcHist([gray_image], [0], 
                              None, [256], [0, 256])

histogram2 = cv2.calcHist([gray_image2], [0], 
                              None, [256], [0, 256])

print("Part 1: Joint histogram - section a")
print(" calculates the joint histogram of two images of the same size")
from matplotlib import pyplot as plt

img = cv2.imread('11.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


img2 = cv2.imread('22.jpg',0)
plt.hist(img2.ravel(),256,[0,256]); plt.show()

sigma_hist_1 = sum(histogram1)
sigma_hist_2 = sum(histogram2)

print("Part 1: Assertion Sigma Histogram = n*p - section b")
print("Prove that images in size 220*180 Sigma Histogram,j(i,j)=n*p")
# Prove that images in size 220*180 Sigma Hi,j(i,j)=n*p
print("Sigma of Fist Histogram= %d & n*p= %d" %(sigma_hist_1, n*p))
print("Sigma of Second Histogram= %d & n*p= %d" %(sigma_hist_2, n*p))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("Part 1: Logarithmic Scale - section c")
print("Visualize joint hist by using the logarithmic scale")
data1=np.array(histogram1)
data2=np.array(histogram2)
data=np.concatenate((data1, data2), axis=1)

df = pd.DataFrame(data)
df = pd.melt(df, var_name='Category')  

g = sns.FacetGrid(df, col='Category', col_wrap=2, sharex=True, sharey=False, aspect=1.5)
g = g.map(plt.hist, "value", color="r")
g.axes[0].set_yscale('log')
g.axes[1].set_yscale('log')
