import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

#cap = cv2.VideoCapture('/home/tsoprano/Documents/UTA/AI_Project/ktmTraff.mp4')
cap = cv2.VideoCapture('/home/tsoprano/Downloads/plates1.mp4')
# if you have multiple webcams change the value to the correct one
#cap = cv2.VideoCapture("1")   # if you have multiple webcams change the value to the correct one
#sprint(cap)

# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'new_graph'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training1', 'object-detection.pbtxt')

NUM_CLASSES = 1


# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.get_default_graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#predict odd/even
NUM_MODEL = tf.keras.models.load_model("/home/tsoprano/Downloads/object_detection/numDet/64x2-CNN-3_64_1.model")
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]
IMG_SHAPE = 50
def prepareAndPredict(image):
	img_array = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	new_array = cv2.resize(img_array, (IMG_SHAPE, IMG_SHAPE)).reshape(-1, IMG_SHAPE, IMG_SHAPE, 1)
	prediction = NUM_MODEL.predict(new_array)
	for index, val in enumerate(prediction[0]):
		if val == 1:
			p = "value: {} . So, number plate is {}".format(CATEGORIES[index], "Even" if int(CATEGORIES[index])%2==0 else "Odd" )
			print(p)


def image_segmentation(coordinates,image):
	height = image.shape[0]
	width = image.shape[1]
	xmin = math.ceil(coordinates[1]*width)
	ymin = math.ceil(coordinates[0]*height)
	xmax = math.ceil(coordinates[3]*width)
	ymax = math.ceil(coordinates[2]*height)
	img = image[int(ymin):int(ymax), int(xmin):int(xmax)]
	
	#from threshold
	img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	retval1, grayThreshold = cv2.threshold(img2gray, 150,255, cv2.THRESH_BINARY)
	gaus = cv2.adaptiveThreshold(grayThreshold,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115,1)
	contours, hierarchy = cv2.findContours(gaus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	#from color
	inv_img = cv2.bitwise_not(img)
	hsv_inv = cv2.cvtColor(inv_img, cv2.COLOR_BGR2HSV)
	lower_cyan = np.array([80,70,50])
	upper_cyan = np.array([100,255,255])
	mask3 = cv2.inRange(hsv_inv, lower_cyan, upper_cyan)
	contours_color, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	
	for c in contours_color: 
		#if cv2.contourArea(c)<10000:
		mask = np.zeros_like(img)
		cv2.drawContours(mask, c, -1, 255, 1) # Draw filled contour in mask
		#cv2.imshow("test",mask)
		#print(cv2.contourArea(c))
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extBot = tuple(c[c[:, :, 1].argmin()][0])
		extTop = tuple(c[c[:, :, 1].argmax()][0])
		image = cv2.rectangle(img, (extLeft[0],extBot[1]), (extRight[0],extTop[1]), (255,0,0), 1)
	
	cnts = sorted(contours,key=cv2.contourArea, reverse = True)[1:10]
	maxRight = 0
	#val = cnts[-1]
	r=[0,0]
	l=[0,0]
	b=[0,0]
	t=[0,0]
	for c in cnts:
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		if (extRight[0]>maxRight):
			maxRight = extRight[0]
			#val = c
			r = extRight
			l = tuple(c[c[:, :, 0].argmin()][0])
			b = tuple(c[c[:, :, 1].argmin()][0])
			t = tuple(c[c[:, :, 1].argmax()][0])

	image = cv2.rectangle(img, (l[0],b[1]), (r[0],t[1]), (255,0,0), 1)
	#image = cv2.rectangle(img, (10,40), (10,40), (255,0,0), 3)
	num_image = image[b[1]:t[1],l[0]:r[0]]
	#prepareAndPredict(num_image)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            # get all boxes from an array
            max_boxes_to_draw = boxes.shape[0]
            # get scores to get a threshold
            scores = np.squeeze(scores)
            # detection %age
            min_score_thresh=.8
            # iterate over all objects found
            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            	#if higher than theshold %age
            	if scores is None or scores[i] > min_score_thresh:
            		# boxes[i] is the box which will be drawn
            		class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
            		image_segmentation(boxes[i],image_np)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('object detection', cv2.resize(image_np, (640, 480)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
