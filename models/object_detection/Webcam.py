
# coding: utf-8

# # Real Time Object Detection
# 
# Welcome to Real Timne Object Detection. This code will walk you step by step through the process of using a pre-trained model to detect objects in real time.

# ## Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# 
# Here are the imports from the object detection module.


from utils import label_map_util

from utils import visualization_utils as vis_util


# ## Model Preparation

# In[3]:
# import pyttsx3
# engine = pyttsx3.init()
from threading import Thread

from pygame import mixer


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('models/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

labels_present=["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
                "parking meter","bench" ,"bird","cat","dog","horse", "sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
                "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
                "clock","vase","scissors","teddy bear","hair drier","toothbrush"]


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
 
# Label maps map indices to category names, so that when our convolution predicts 5, we know that this corresponds to `airplane`. 



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Start Webcam




# Initialize webcam
cap = cv2.VideoCapture(0)


# ## Detection

# In[8]:

object_count=0
display_str="None"
object_close=0
object_far=0



def detect():
    mixer.init()
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            ret = True
            
            while (ret):
                ret,image_np = cap.read()

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
            
            
                classes1=np.squeeze(classes).astype(np.int32)
                
                global object_count
                global display_str 
                for i in range(min(20, np.squeeze(boxes.shape[0]))):
                    if classes1[i] in category_index.keys():
                        class_name = category_index[classes1[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                if display_str!= "N/A":
                    object_count+=1
                if object_count>35:
                    if display_str in labels_present:
                        mixer.music.load(f'models/object_detection/audios/{display_str}.mp3')
                        mixer.music.play()
                        object_count=0
                
                
             

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                

                for i,b in enumerate(boxes[0]):
                    
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                        # mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                       
                              
                
                    global object_far
                    global object_close
                    if apx_distance <= 0.1:
                        if mid_x > 0.3 and mid_x < 0.7:
                            object_close+=1
                            object_far=0
                            if object_close>6000:
                                mixer.music.load(f'models/object_detection/audios/ObjectClose.mp3')
                                mixer.music.play()
                                object_close=0
                                object_far=0
                       
                    else:
                        if mid_x > 0.3 and mid_x < 0.7:
                            object_close=0
                            object_far+=1
                            if object_far>6000:
                                mixer.music.load(f'models/object_detection/audios/ObjectSafe.mp3')
                                mixer.music.play()
                                object_close=0
                                object_far=0
                        
                cv2.imshow('image',cv2.resize(image_np,(720,480)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    break
                    
detect()