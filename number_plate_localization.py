import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import pytesseract


print(cv2.__version__)

def plot_image(img1, img2, title1="", title2=""):
	fig = plt.figure(figsize=[15,15])
	ax1 = fig.add_subplot(121)
	ax1.imshow(img1, cmap='gray')
	ax1.set(xticks=[], yticks=[], title=title1)
	
	ax2 = fig.add_subplot(122)
	ax2.imshow(img2, cmap='gray')
	ax2.set(xticks=[], yticks=[], title=title2)
	
	
	
	
	
path = "./images/car_2.jpg"

image = cv2.imread(path)

plot_image(image, image)
plt.imshow(plt.imread(path))
plt.savefig(path)
plt.show()
