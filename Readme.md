# Project Core50 image segmentation

In this project, I used the CORE50 dataset to compose, train and test an image segmentation algorithm to detect the object "mobile phone". 
This convolutional neural network takes an RGB image and spits out a black&white image: the area where the mobile phone is will be in white and the rest should be black.


### Summary:
I. Implementing a pretrained model: Mask RCNN

II. Train the model on CORE50 'mobile phone' photos

III. Building test script to test the model on webcam




### I. Implementing a pretrained model: Mask RCNN

Firstly, instead of training a model from scratch, transfer learning allowed me to start with a weights file that contains weight values of a CNN that was previously trained on the COCO dataset. The COCO dataset contains many images (~120K), so the trained weights have already learned many features common in natural images. In this case, Mask R-CNN (regional convolutional neural network) is a two stage framework: the first stage scans the image and generates proposals(areas likely to contain an object). The second stage classifies the proposals and generates bounding boxes and masks. 


The pre-trained weights (trained on MS COCO) file "mask_rcnn_balloon.h5" was too large and I couldn't upload it to the github repository. For the program to work, the file need to be downloaded at this link (https://github.com/matterport/Mask_RCNN/releases) and pasted in the samples folder of the Mask_RCNN repository.

Fortunately, the coco dataset has a "cell phone" class, so Mask R-CNN was already capable to classify, detect and segmentate cell phones in images.

Some image segmentation results on Core50 are showed below:

<img src ="image_segmentation_results/C_01_06_001_segmentation.PNG" width = "250"> <img src ="image_segmentation_results/C_01_06_001_Blackandwhite.PNG" width = "250">

<img src ="image_segmentation_results/C_01_07_000_segmentation.PNG" width = "250"> <img src ="image_segmentation_results/C_01_07_000_Blackandwhite.PNG" width = "250">

<img src ="image_segmentation_results/C_01_08_000_segmentation.PNG" width = "250"> <img src ="image_segmentation_results/C_01_08_000_Blackandwhite.PNG" width = "250">

<img src ="image_segmentation_results/C_01_09_002_segmentation.PNG" width = "250"> <img src ="image_segmentation_results/C_01_09_002_Blackandwhite.PNG" width = "250">


More details on the code implementing Mask R-CNN are in the notebook file: "\Mask_RCNN\samples\mrcnn.ipynb"

Note: The setup of Mask R-CNN took quite a long time, this is not only because of the libraries that need to be used, but also because some libraries required specific older versions. For instance, I needed to downgrade tensorflow and Keras to older versions.

It is more common to be in situations in which we would like to segmentate objects that Mask R-CNN or any public network has never been trained to. In this case, we would need to train the network to learn how to segmentate specific class of objects. 

### II. CORE50 mobile phone training photos

Let's assume now that Mask R-CNN has never been trained on cell phones, how can we train this network on Core50 mobile phone photos?
There is an excellent resource explaining how this can be done by matterport here (https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46).

The article explains that they would like to train the Mask R-CNN to recognize "balloons", which is an object class that Mask R-CNN has never learned to detect in the past. 
For the network to learn to detect balloons, we need to train a network to minimize its loss function over the learning process. 
In image segmentation, mIoU (mean interception over union) is a common metric used to calculate the loss function. The IoU is the ratio between the area of overlap and the area of union between the ground truth and the predicted areas. Image pixel annotation is crucial for image segmentation learning process since it is used to calculate the mIoU. 
However, the Core 50 dataset does not contain pixel annotation for the objects in the images. Therefore, there are several options that can be considered to train the network.

##### II. 1) Depth images of the Core50 dataset.

Although the core 50 dataset does not contain the pixel annotation of the object in its images, the Core50 dataset does contain image depth information, which can easily help users to extract and to separate background from the object and the hand holding the object. However, we want to train a network that is capable of detecting and segmenting mobile phone in any images even when the depth information of the images is not available.

First, an algorithm has been used to discard all pixels belonging to the background. Then, an SVM classifier is used to discard the user's hand from the image to only keep the object's pixel in the image. The process is explained in this Github: https://github.com/giacomobartoli/core50_segmentation

<img src ="annotations/image_segmentation_CORE50.PNG" width = "800">

This can be a pretty efficient way to annotate the images. Unfortunately, the image segmentation results from this process is quite approximate. If we use these results as a the reference to compute the mIoU, the results will only be even more approximate.

##### II. 2) Manual pixel annotation
The author downloaded 75 photos of balloons on Flickr and annotate manualy the balloons area on the images with VGG Image Annotator (VIA) (http://www.robots.ox.ac.uk/~vgg/software/via/). 
In our case, Core50 has 11 sections of short videos of 50 objects. For mobile phones, there are 5 mobile phones in each section and for each phone there are around 300 frames that were recorded. Therefore, there are in total 11 * 5 * 300 = 16500 photos of mobile phone in Core50 dataset. Due to the time constraints, the pixel annotation on every image was not a feasible option, but here are some annotated images for illustration purposes:

<img src ="annotations/C_01_06_001.PNG" width = "300"> <img src ="annotations/C_01_07_000.PNG" width = "300">

<img src ="annotations/C_01_08_000.PNG" width = "300"> <img src ="annotations/C_01_09_002.PNG" width = "300">

Once these photos annotated, they have been exported in a JSON file, and can be further used to train the network.
In reality, the image annotation process can be completed using online collaborative platforms, such as Labelbox(www.labelbox.com).


### III. Building test script to test the model on webcam
To build the test script to test the model on a webcam, the author of the following article explained how to do a real time image segmentation using Mask R-CNN: https://www.akshatvasistha.com/2019/10/how-do-real-time-image-segmentation-mask-rcnn.html
After modifying the real time segmentation python file, the notebook file 'Real_time_segment.ipynb' has been created in order to run the real time image segmentation file on a computer's webcam.


### Conclusion:

In this assignment, I started by exploring the popular image segmentation networks Mask R-CNN that was pre-trained on the popular COCO dataset. Fortunately, the COCO dataset contains mobile phone pictures, and Mask R-CNN was therefore capable of completing image segmentation on mobile phone. Then, making the assumption that Mask R-CNN was never  trained on mobile phone pictures, I presented several options that could have been taken to learn to make segmentation on cell phones. Annotating manually the images is the most time-consuming but also the most realistic way to make pixel-level image annotation. Once the images annotated, it would be possible to train the network to recognize new objects. Finally, a real time segmentation python file was composed to test the network on webcams.


### Resources:

1) CORe 50 dataset: https://vlomonaco.github.io/core50/index.html#dataset

2) Mask RCNN: https://github.com/matterport/Mask_RCNN/releases

3) Implementation of Mask R-CNN: https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/?utm_source=blog&utm_medium=introduction-image-segmentation-techniques-python

4) Training Mask R-CNN on new objects: https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

5) Real time image segmentation with Mask R-CNN: https://www.akshatvasistha.com/2019/10/how-do-real-time-image-segmentation-mask-rcnn.html

6) VGG image annotator: http://www.robots.ox.ac.uk/~vgg/software/via/

7) Augmenting Gastrointestinal Health: A Deep Learning Approach to Human Stool Recognition and Characterization in Macroscopic Images: https://arxiv.org/ftp/arxiv/papers/1903/1903.10578.pdf
