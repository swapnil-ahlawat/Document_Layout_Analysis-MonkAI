# Document Layout Detection using MonkAI Object Detection Library

Models that takes a document image file as input and locates the position of a set of objects, predicts their labels and confidence scores.

## Data Preprocessing 
### (Data_Preprocess+Selective_Data_Augmentation+VOC_to_Monk_Type.ipynb)

- **Normalisation:** Calculated Mean & Standard deviation of training images (3 images taken out of dataset for inference) to feed into model for normalisation (used in FasterRCNN).

- **Format Conversion:** TIFF format was causing problems in data augmentation and training on TIFF images was more than 5x slower than JPEG format images because of their huge size. Therefore, TIFF images were converted to JPEG format images.

- **Selective Data Augmentation:** In the raw dataset 4750+ paragraph type objects and only 10-30 frames, graphics, etc. type objects which led to huge bias in dataset. To generate more data and to decrease bias, a customised function has been implemented from the scratch. This function produces random translational augmentated images of only those images which have minority classes in them. Using this function, the dataset size increased from 475 images to 1783 images. If data augmentation has been done on every image, there would have been 24000+ paragraphs in the dataset whereas there are 19568 now, which slightly improves the bias. (Exact numbers are in the data preprocessing notebook). The augmented images had to be saved in the dataset because augmentation function couldn't be called on the go. 

- **Conversion to VOC to Monk type-** Coversion so that Monk format can be later used in SSD Model, converted to yolo type for Yolo model, and COCO format for FasterRCNN Model.



# Choice of architecture
-Inspiration from the blog- https://medium.com/@Intellica.AI/a-comparative-study-of-custom-object-detection-algorithms-9e7ddf6e765e

Yolov3, FasterRCNN & SSD are broadly top 3 model architectures that are used for Object detection. So, for this task, prediction and confidence on inference images of these 3 architectures have been compared.


## Yolov3 
### (YOLOv3-Document_Analysis.ipynb)
Yolov3 pipeline of Monk Object Detection Library has been used for implementing this model.
Firstly, the dataset was converted to Yolo format. The mode was trained for 10 epochs, with batch size=8, learning rate=0.005 and "sgd" optimizer ("sgd" performed better than "adam" optimizer). It achieved F1 score of 0.348 and mAP@0.5 of 0.318.

Following are the results the model achieved on some test images:

<img src="/Output_Images/yolov3_test1.jpeg" width="350" />
<img src="/Output_Images/yolov3_test2.jpeg" width="350" />
<img src="/Output_Images/yolov3_test3.jpeg" width="350" />

The model is predicting most of the labels correctly with high confidence (though it is missing some of them) and the bounding boxes are also fine.
Both of them can be greatly improved by training for more epochs. The training curves (in the notebook) haven't flattened out yet which suggests that the model can be trained more.


## FasterRCNN with VGG16 Backend 
### (FasterRCNN-VGG16_Backend-Document_Analysis.ipynb)
MXRCNN pipeline of Monk Object Detection Library has been used for implementing this model.
After some comparisons, it was found out that VGG16 performs better than ResNet101 for object detection task using FasterRCNN, so VGG16 is chosen for the backend. Firstly, the dataset was converted from Monk Format to COCO Format. The model was trained for 3 epochs with learning rate of 0.005 and then for 3 more epochs with learning rate of 0.001. The image was preprocessed to smaller size (min 300px, max 500px) and normalsied using mean and standard deviation calculated in preprocessing notebook. Batch size has been kept at 2 because more than that was causing CUDAOutOfMemory error. It achieved RPN Accuracy=0.807714 and RCNN Accuracy= 0.750662.

Following are the results it achieved on some test images:

<img src="/Output_Images/fasterRCNN_test1.png" width="350" />
<img src="/Output_Images/fasterRCNN_test2.png" width="350" />
<img src="/Output_Images/fasterRCNN_test3.png" width="350" />

As it can be seen, the model is performing very poorly on test images. The reason for this issue is still unknown (can be issue in conversion to COCO format, pause and resume issue or backend issue) as the model was taking a lot of time in training (45 minutes per epoch) and I couldn't explore much due to time constraints.


## SDD with VGG16 Backend and Atrous Convolutions 
### (SSD512-VGG16-Document_Analysis.ipynb)
GluonCV_finetune pipeline of Monk Object Detection Library has been used for implementing this model.
VGG16 has been used here too for comparison purposes. Out of options available for VGG16, "ssd_512_vgg16_atrous_coco" as it was pretrained on COCO dataset which is huge in comparison to VOC dataset. Batch size has been kept at 2 because more than that was causing CUDAOutOfMemory error and learning rate is kept at 0.001 becuase 0.003 was resulting into NaN values. After training for 5 epochs, the model had CrossEntropy loss of 2.680 and SmoothL1 loss=3.254.

Following are the results it achieved on some test images:

<img src="/Output_Images/ssd512_test1.png" width="350" />
<img src="/Output_Images/ssd512_test2.png" width="350" />
<img src="/Output_Images/ssd512_test3.png" width="350" />

The model is performing well in identifying objects with very high confidence but it is biased a lot towards paragraphs. Its performance can be improved by using bigger batch size, training for more epochs  and more data augmentation techniques to reduce bias.


## Inference
On the basis of training done till now, SSD architecture with VGG16 backend is giving far better than other two architectures. It is producing accuracte bounding boxes with very high confidence. There is issue with the classification task which is due to bias in the dataset. The models performance can be greatly improved by using bigger batch_size (currently ==2), training for more epochs (trained for only 5 epochs for now) and using different methods to reduce bias in the dataset.

## Limitations & Future Improvements
- The models have been trained for only 5-10 epochs. They can be trained for more epochs for better accuracy.
- Data has been unbiased by translational data augnmentation only. More data augmentation techniques (such as rotation and random cropping) and other techniques can be applied to generate more data with more number of minority classes/ reduce bias.
- Batch size of SSD and FasterRCNN has been kept equal to 2 in the following training due to logistics limitation which is an issue in generalization. The model can be trained with batch size >=2 for better generalisation.
- There's a slight possibility of data handling issue in FasterRCNN model as the predictions of output images is really bad. Some work can be done on that to ensure there is no issue in data conversion.

