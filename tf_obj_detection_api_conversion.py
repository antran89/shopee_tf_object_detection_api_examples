# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:52:39 2017

Trials of Shopee datasets with Faster RCNN implemented in TF Object Detection API
Convert Shopee fashion datasets into detected results using TF API.

@author: tranlaman
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
TF_API="/media/tranlaman/data/projects/object_dection/models/research/object_detection"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from utils import label_map_util
from utils import visualization_utils as vis_util
import argparse
import glob

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection for Shopee dataset.')
    parser.add_argument('--input_dataset', dest='input_dataset', help='input dataset folder.', required=True,
                        type=str)
    parser.add_argument('--result_dataset', dest='result_dataset', help='result dataset folder.', required=True,
                        type=str)
    parser.add_argument('--model_name', dest='model_name', help='object detection model name.', 
                        default='faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017',
                        type=str)
    parser.add_argument('--image_ext', dest='image_ext', help='image extentions.', default='jpg',
                        type=str)

    args = parser.parse_args()

    return args
    
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

def main():

    args = parse_args()
    input_dataset = args.input_dataset
    result_dataset = args.result_dataset
    MODEL_NAME = args.model_name
    image_ext = args.image_ext
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(TF_API, MODEL_NAME, 'frozen_inference_graph.pb')
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(TF_API, 'data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90
    
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    # helper code
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
          
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    TEST_IMAGES = glob.glob(os.path.join(input_dataset, '*.%s' % image_ext))
    TEST_IMAGES.sort()
         
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        for image_path in TEST_IMAGES:
          image = Image.open(image_path)
          image_file_name = os.path.basename(image_path)
          result_image_path = os.path.join(result_dataset, image_file_name)
          if os.path.isfile(result_image_path):
              continue
          
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          
          # save the result image
          result_img = Image.fromarray(image_np)
          result_img.save(result_image_path)
          
if __name__ =='__main__':
    main()
