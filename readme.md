# Background

MonReader is a new mobile document digitalization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly. It also dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally, recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor. 

## Data Description

Dataset is a collection of page flipping video from smart phones and labelled them as flipping and not flipping. 

The videos are clipped as short videos and labelled as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

# Goal
Predict if the page is being flipped using a single image. 
Next goal is to predict if a given sequence of images contains an action of flipping (Bulk)

## Success Metrics
Evaluate model performance based on F1 score, the higher the better

# Data Exploration

## Image Resolution
All the images are in 1920 x 1080 (1080P) resolution.

## Data Size
For training, there are 2392 training dataset images. For testing there are 597 images.

# Data Preparation

## Define a Validation set from the Training Set
20% of the training data is split into a validation set. Since the main goal is to predict if one page is being flipped or not, the order of the images being split does not matter.
This data is loaded into Tensorflow's Dataset class.

## Set Image Batch Size
One Batch is 32 Images

## Set Image Size to Load into the Dataset
From the original image's resolution of 1920 x 1080P, the dataset reads the images with the size of 180x180.


# Training

## f-1 score Metric
Metrics are for logging the training process.
Since by default there is no f1 score as a metric, the f1 score function is developed with keras backend and added as a 
training metric.

## Model 1: Basic CNN (Baseline)

This Sequential model consists of three convolution blocks (tf.keras.layers.Conv2D) with a max pooling layer (tf.keras.layers.MaxPooling2D) in each of them. There's a fully-connected layer (tf.keras.layers.Dense) with 128 units in the end, that is activated by a ReLU activation function ('relu').

### Details
Optimizer: Adam. Adam is used because it is generally a good gradient descent algorithm that works well for most problems.
Loss function: Cross Entropy loss is used since this is a classification problem.
Metric: Accuracy and F-1 score are both tracked.
The setup above will be the general setup for the subsequent training processes.

Number of epochs: 30
The number of epochs was set to 30, but training seems to stabilize at around the 10th epoch.


### Results
Training Validation accuracy: 99%
Training F1 score: 0.6758

Testing accuracy: 0.9916
Testing F1 score: 0.6856

In the logs of the training process, it seems that after epoch 25, the validation accuracy is decreasing. There might be overfitting, which will be tackled on the second model. 

## Model 2: CNN with regularization and dropout on the Dense Layers

To combat overfitting, both L2 and dropout are added to the dense layer.

### Details
Setup is similar to the previous Model.

### Results
Training Validation accuracy: 99%
Training F1 score: 0.6751

Testing accuracy: 0.9899
Testing F1 score: 0.6724

The model performed worse on the testing set compared to the first model. 

## Model 3: Feature Extraction with MobileNetV2

Adapted from: https://www.tensorflow.org/tutorials/images/transfer_learning#create_the_base_model_from_the_pre-trained_convnets


Here, we will use the representations learned by a previous network to extract meaningful features from new samples. 
This will be done by adding a new classifier, which will be trained from scratch, on top of the pretrained model so that we can repurpose the feature maps learned previously for the dataset.

We don't need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.


### MobileNetV2
This Model from Google is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. 

We instantiated a MobileNetV2 model pre-loaded with weights trained on ImageNet. Since we don't need the classification layers at the top, we use `include_top` = False. 

### Freeze Convolutional Base Layers

Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training.

Many models contain tf.keras.layers.BatchNormalization layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial.

When we set layer.trainable = False, the BatchNormalization layer will run in inference mode, and will not update its mean and variance statistics.

### Add Classification Head
Here, we add a pooling2D layer to convert the features to a single 1280-element vector per image, and Dense layer to convert  these features into a single prediction per image.


### Results
Testing Accuracy: 0.79906 
Testing F-1 Score: 0.7729

With a lower accuracy, the model beats both previous models.



## Model 4: Fine Tune MobileNetV2

To increase performance even further is to train (fine-tune) the weights of the top layers of the pre-trained model alongside the training of the classifier added in the previous model. The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.


In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features that generalize to almost all types of images. The higher the layers, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

### Un-Freeze Top layers of the model 
From layer 100 onwards, the base model is unfreezed.


### Lower Learning Rate
As we are training a much larger model and want to readapt the pretrained weights, it is important to use a lower learning rate at this stage. Otherwise, the model could overfit very quickly.

Thus, we use Model 3's learning rate divided by 10.

### Results
Testing Accuracy: 0.9933
Testing F-1 Score: 0.9937

This model performs the best out of all the previous ones.









