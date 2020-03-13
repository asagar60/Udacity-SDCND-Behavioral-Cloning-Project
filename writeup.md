# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/model_actitecture.JPG "Model Architecture"
[image2]: ./writeup_images/flipped_image.JPEG "Flip Image"
[image3]: ./writeup_images/translated_image.JPEG "Translated Image"
[image4]: ./writeup_images/zoomed_image.JPEG "Zoomed Image"
[image5]: ./writeup_images/rotated_image.JPEG "Rotated Image"
[image6]: ./writeup_images/YUV_image.JPEG "YUV image"
[image7]: ./writeup_images/loss_graph.JPEG "Loss Graph"
[image8]: ./writeup_images/Train_network.JPEG "Training Strategy"
[image9]: ./writeup_images/simaulator_testing.JPG "Simulator Testing"
[image10]: ./writeup_images/center.JPG "Center Image"
[image11]: ./writeup_images/left.JPG "Left Image"
[image12]: ./writeup_images/right.JPG "Right Image"
[image13]: ./writeup_images/YUV_image_1.JPEG "Color Conversion"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavior Cloning Project v2.0.py containing the script to create and train the model
* Behavior Cloning Project v2.0.ipynb file for visualization of code cells
* Behavior Cloning Project v2.0.html file
* drive.py for driving the car in autonomous mode
* drive_updated.py for preprocessing the input images and driving the car in autonomous mode.
* model_v2.0.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive_updated.py model_v2.0.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The approach for model design is from  DAVE-2 system from the paper 'End to End Learning for Self-Driving Cars' from NVIDIA , published on 25 Apr 2016.

The model consists of 9 layers. 1 Lambda Layer ( Normalization Layer ), 5 Convolution Layer, 3 Fully Connected Layer.
My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (code cell 43 of Behavior Cloning Project v2.0.ipynb).
The model includes RELU activation function to introduce nonlinearity (code cell 43 of Behavior Cloning Project v2.0.ipynb), and the data is normalized in the model using a Keras lambda layer (code cell 31 of Behavior Cloning Project v2.0.ipynb).



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with dropout probability of 0.5 in order to reduce overfitting (code cell 43 of Behavior Cloning Project v2.0.ipynb).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code cell 43 of Behavior Cloning Project v2.0.ipynb). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, with manual tuning of learning rate to achieve better results.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Following data were used:

Centre Lane Driving :

a. Track 1

b. Track 2

c. Track 1 reverse

d. Track 2 reverse

I used a combination of center lane driving along with center lane driving in opposite direction.

For recovering from the left and right sides of the road, I used images from left and right camera so that the network would learn how to recover from mistakes.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep the model short as possible and achieve minimum mean squared error.

My first step was to use a convolution neural network model similar to the DAVE-2 model proposed by NVIDIA in their paper. I thought this model might be appropriate because it was designed based on the idea that with minimum training data the system can learn to drive in traffic on local roads.

![Training Strategy][image8]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added multiple dropout layers so that the network would still learn in absence of forward links , and thus have better learning .

Then I adjusted learning from 0.001 to 0.0001 .This significantly improved the model and therefor reduced overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. There were instances when the car went out of the lane but then itself came back to the center lane .

Since, this was not an optimal user  scenario and would cause tremendous problems for the driver, so to improve the driving behavior in these cases, I increased the EPOCHS to learn better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (code cell 43 of Behavior Cloning Project v2.0.ipynb) consisted of a convolution neural network with the following layers and layer sizes .

Brief Description of the model.


|          Layer        |              Description	        		                |
|:---------------------:|:---------------------------------------------------------:|
| Input         		| 66x200x3 YUV image   					                    |
| Lambda layer          | Normalizing the image (mean 0 , range [-1,1])             |
| Convolution 5x5     	| maxpooling: 2x2, activation: 'relu'  output: 31x98x24 	|
| Convolution 5x5     	| maxpooling: 2x2, activation: 'relu'  output: 14x47x36 	|
| Convolution 5x5     	| maxpooling: 2x2, activation: 'relu'  output:  5x22x48 	|
| Convolution 5x5     	| maxpooling: 2x2, activation: 'relu'  output:  3x20x64 	|
| Convolution 5x5     	| maxpooling: 2x2, activation: 'relu'  output:  1x18x64 	|
| Dropout          	    | rate = 0.5      								            |
| Flatten               | outputs 1152                                              |
| Fully connected       | activation: 'relu' output: 100                            |
| Dropout          	    | rate = 0.5      								            |
| Fully connected       | activation: 'relu' output: 50                             |
| Fully connected       | activation: 'relu' output: 10                             |
| output layer          | outputs 1                                                 |



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![DAVE-2][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track-1 using center lane driving. Here is an example image of center lane driving:

![Center Image][image10]

I then recorded 3 laps driving in opposite direction . This would not only increase the data to train on, but also makes the model unbiased as the track-1 has majority of left turns and few right turns.

I have used images from left and right camera , with tweaked steering angle .

This has 2 advantages :
    a. Our model will learn how to recover from the side track and come back to center lane.
    b. 3 times the data

Here are the images from left , center and right cameras

![Left Image][image11]  ![Center Image][image10] ![Right Image][image12]

Then I repeated this process on track -2 in order to get more data points.

After the collection process , I randomly shuffled the data set and put 20% of the data into a validation set.

Then I used a data generator to shuffle , preprocess, Augment and yield batch of training and validation samples on the fly.

_Note_: Because I didn't want to augment my validation data , I used an extra variable which decides, if the data needs to be augmented or not

##### Preprocessing the Data
I preprocessed the data by applying following techniques:

1. I cropped the less useful data, i.e. upper portion with trees and forest, and lower portion with car's hood.
2. Converting the image from RGB color space to YUV color space.
3. Resize the image to 200x66x3

YUV image
![YUV image][image13]

Preprocessed Image
![preprocessed Image][image6]

_Note_ : YUV color space and Image size with 200x66x3 is recommended in the paper . Which gave amazing results and help generalize the model.

##### Image Augmentation

Then I Augmented the data. after trying different image augmentations , I finally stick to 3 data augmentations :

a. Random Image flip ( steering angle will be changed and will be negative of what was before)

![flipped image][image2]

b. Random Image shift (Images will either be left/ right shift or up/ down shift i.e. Translation)

![translated image][image3]

c. Zoomed Image (factor 1.x to 1.3x)

![Zoomed image][image4]

d. Random Image rotation ( -25 deg to +25 deg)

![Rotated Image][image5]    

These Augmentations were recommended in the paper 'End to End Learning for Self-Driving Cars' from NVIDIA. (para 5.2)

##### Training and Validating the Data

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30.  I used an adam optimizer with learning_rate = 0.0001.

![Loss Graph][image7]


#### Simulator Testing

Now that we have trained our model , and generalized it , so that it can run on both tracks , we will run on the simulator .

![Simulator Testing][image9]

we will only use the image from camera mounted on the center to calculate the steering angle.

_Note_: I have updated the drive.py file , so that it can preprocess the images before sending it to the model. I have included both drive.py and drive_updated.py

To run the model ,

```sh
python drive_updated.py model_v2.0.h5
```


### Output Video

Track 1:

output of track 1 is [here](Submission_1/track1_output.mp4)

Track 2

output of track 2 is [here](Submission_1/track2_output.mp4)
