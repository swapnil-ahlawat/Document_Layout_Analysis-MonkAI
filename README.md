# Document_Layout_Detection-MonkAI
Models that takes a document image file as input and locates the position of a set of objects, predicts their labels and confidence scores.

## Data-Preprocessing (Data_Preprocess+Selective_Data_Augmentation+VOC_to_Monk_Type.ipynb)
- Normalisation: Calculated Mean & Standard deviation of training images (3 taken out for inference) to feed into model for normalisation.
- Format Conversion: TIFF format was causing problems in data augmentation and training on TIFF was more than 5x slower than JPEG format images because of their huge size. Therefore TIFF images are converted to JPEG format images.
- Selective Data Augmentation: In the raw dataset 4750+ paragraphe type objects and only 10-30 frames, graphics, etc type objects which led to huge bias in dataset. To generate more data and to decrease bias, a customised function from the scratch has been implemented. This function produces random translational augmentated images of only those images which have minority classes in them. Using this function the dataset size increased from 475 images to 1783 images. If data augmentation has been done on every image, there would have been 24000+ paragraphs in the dataset whereas there are 19568 now, which slightly improves the bias. (Exact numbers in the data preprocessing notebook)
- Conversion to VOC to Monk type- So that it can be later used in SSD Model, converted to yolo type for Yolo model, and COCO format for FasterRCNN Model.

## Choice of architecture
-Inspiration from the blog- https://medium.com/@Intellica.AI/a-comparative-study-of-custom-object-detection-algorithms-9e7ddf6e765e

Yolov3, FasterRCNN & SSD are the top 3 model architectures that are used in Object detection. So, for this task, inference prediction and confidence of these 3 architectures have been compared.

## Yolov3 (YOLOv3-Document_Analysis.ipynb)
First of all, the dataset was converted to Yolo type format. Then it was trained for 10 epochs with learning rate=005 and sgd optimiser. It achieved F1 score= 0.348 and mAP@0.5=0.318.
Following are the results the model achieved on some test images:
![GitHub Logo](/images/logo.png)
![GitHub Logo](/images/logo.png)
![GitHub Logo](/images/logo.png)

## FasterRCNN with VGG16 Backend (FasterRCNN-VGG16_Backend-Document_Analysis.ipynb)
After some research, it was found out that VGG16 performs better than ResNet101 for object detection task, so VGG16 is chosen for the backend. The model was trained for 6 epochs and it achieved RPN Accuracy=0.807714 and RCNN Accuracy= 0.750662. There was some issue with the model which couldn't be resolved due to very high training time (45 minutes per epoch) and overall time constraint.
Following are the results it achieved on some test images:
![GitHub Logo](/images/logo.png)
![GitHub Logo](/images/logo.png)
![GitHub Logo](/images/logo.png)

## SDD with VGG16 Backend and Atrous Convolutions (SSD512-VGG16-Document_Analysis.ipynb)
VGG16 was used here too for comparison.
