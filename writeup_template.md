**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/figure_1.png "Loss Visualization"
[image3]: ./examples/counter_clockwise.jpg "Counter clockwise driving"
[image4]: ./examples/clockwise.jpg "Clockwise driving"

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The Nvidia network architecture was used in this project. The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. (model.py lines 58-69) 

The convolutional layers consists of three convolutional layers with a 2x2 stride and a 5x5 kernel and a non-strided convolution
with a 3x3 kernel size in the last two convolutional layers.

The data is normalized in the model using a Keras lambda layer (code line 58) and to ensure that the model trains to learn only the lane features the input image was cropped to reomve unnecessary features using the Keras Cropping2D layer (code line 59). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18-26). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

The training data consists of two datasets with driving data of two laps each. The first dataset consists of driving in clockwise direction and the second in counter-clockwise direction. 

#### 5. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added more data to the training set. Initially I used only counter-clockwise driving dataset. Then I added dataset of clockwise driving. This helped generalise the model and helped predict steering angles better. Another approach used to improve steering predictions was to use mouse to drive the car in the simulator rather than arrow keys.

Here is the visualization of losses on both training and validation datasets.

![alt text][image2]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track during turnings and unable to identify lane where one part of the lane line was missing and the car would drift away off-road. To improve the driving behavior in these cases, I used datasets with smooth transitions in curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a visualization of the architecture

![alt text][image1]

To augment the data sat, instead of flipping the images I used clockwise driving datasets of two laps to help generalise the model.

![alt text][image3]
![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
