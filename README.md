# Tensorflow Object Detection API for Shopee fashion dataset
We evalute Faster R-CNN in Tensorflow Object Detection API for Shopee fashion dataset. The details of the API can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Table of contents
Before You Start:
* <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>Installation Tensorflow Object Detection API</a><br>

Setup:
* Changing variable `TF_API` in `tf_obj_detection_api_conversion.py` or `tf_obj_detection_api.py` to the directory where TF Object Detection API is installed. 

Downloading pretrained models:
* Download models into folder `research/object_detection/` of TensorFlow Models folder.
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
tar -xzvf faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
```

Running:
```bash
python tf_obj_detection_api_conversion.py --input_dataset=test_images/ --result_dataset=test_results/
```

Examples:
<p align="center">
  <img src="test_results/img04.jpg" width=676 height=450>
</p>
