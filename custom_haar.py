import urllib.request
import cv2
import numpy as np
import os
import shutil

raw_neg_img_path = 'neg\\'
raw_pos_img_path = 'pos\\'

def store_raw_images():
    pic_num = 1
    # src_files = os.listdir('maize\\')
    # for img in src_files:
        # print(img)
        # shutil.copy('maize\\'+img,'neg\\')
        # os.rename(raw_neg_img_path+img,raw_neg_img_path+str(pic_num))
        # pic_num = pic_num+1
	src_files = os.listdir('111/')
	for img in src_files:
      imgg = cv2.imread('111/'+img, cv2.IMREAD_GRAYSCALE)
      resized_img = cv2.resize(imgg, (50,25))
      cv2.imwrite(raw_pos_img_path+img, resized_img)
      os.rename(raw_neg_img_path+img,str(pic_num)
	
	# src_files = os.listdir(raw_neg_img_path)
	# for img in src_files:
		# imgg = cv2.imread(raw_neg_img_path+img, cv2.IMREAD_GRAYSCALE)
		# cv2.imwrite(raw_neg_img_path+img, imgg)

store_raw_images()

# def create_pos_n_neg():
	# for file_type in ['neg','pos']:
		# for img in os.listdir(file_type):
			# if file_type == 'neg':
				# line = file_type+'/'+img+'\n'
				# with open('bg.txt', 'a') as f:
					# f.write(line)
					
			# elif file_type == 'pos':
				# line = file_type+'/'+img+ ' 1 0 0 50 25\n'
				# with open('info.dat', 'a') as f:
					# f.write(line)
			 
# create_pos_n_neg()


		
		
		
	 
	




  
