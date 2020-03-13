# Project: Build Behavioral Cloning Model
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this Project , we train a Deep Neural Network to learn how to drive a car like human.

Overview
-------
In this project , we will train a deep neural network to simulate human driving behavior.

[//]: # (Image References)

[image1]: ./writeup_images/model_actitecture.JPG "Model Architecture"
[image2]: ./writeup_images/flipped_image.JPEG "Flip Image"
[image3]: ./writeup_images/translated_image.JPEG "Translated Image"
[image4]: ./writeup_images/zoomed_image.JPEG "Zoomed Image"
[image5]: ./writeup_images/rotated_image.JPEG "Rotated Image"
[image6]: ./writeup_images/YUV_image.JPEG "YUV image"
[image7]: ./writeup_images/loss_graph.JPEG "Loss Graph"
[image8]: ./writeup_images/Train_network.JPG "Training Strategy"
[image9]: ./writeup_images/simaulator_testing.JPG "Simulator Testing"
[image10]: ./writeup_images/center.jpg "Center Image"
[image11]: ./writeup_images/left.jpg "Left Image"
[image12]: ./writeup_images/right.jpg "Right Image"
[image13]: ./writeup_images/YUV_image_1.JPEG "Color Conversion"
[image14]: ./readme_images/simulator.JPG "simulator"

**What is Behavioral Cloning ?**
----
Behavioral cloning is a method by which human sub-cognitive skills can be captured and reproduced in a computer program. As the human subject performs the skill, his or her actions are recorded along with the situation that gave rise to the action. A log of these records is used as input to a learning program. The learning program outputs a set of rules that reproduce the skilled behavior. This method can be used to construct automatic control systems for complex tasks for which classical control theory is inadequate.

The goal of this project is to teach a Convolutional Neural Network (CNN) to drive a car in a Udacity simulator.

#### Final Output

![track_1_output](./readme_images/track1_output.gif)
![track_2_output](./readme_images/track2_output.gif)

**Installing Dependencies**
---

- opencv           -  `pip install opencv-python`
- pandas           - `pip install pandas`
- Tensorflow - GPU - `conda install tensorflow-gpu`
- matplotlib       - `pip install matplotlib`
- moviepy          - `pip install moviepy==1.0.0`
- imgaug           - `conda install -c conda-forge imgaug`
- scikit learn     - `conda install -c anaconda scikit-learn`
- keras            - `conda install -c conda-forge keras`
- Flask            - `conda install -c conda-forge flask-socketio`
- Pillow           - `conda install -c anaconda pillow`
- socketio         - `conda install -c conda-forge python-socketio`
- h5py             - `conda install -c anaconda h5py`

### Simulator Details:

![Simulator][image14]

The car is equipped with three cameras that provide video streams and records the values of the steering angle, speed, throttle and brake. The steering angle is the only thing that needs to be predicted, but more advanced models might also want to predict throttle and brake. This turns out to be a regression task. We will use CNN for feature extraction and turn it into a regression model.

github repo : https://github.com/udacity/self-driving-car-sim

## Model Architecture and Training Strategy
----
The approach for model design is from  DAVE-2 system from the paper 'End to End Learning for Self-Driving Cars' from NVIDIA , published on 25 Apr 2016.

http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![DAVE-2][image1]

This model is appropriate because it was designed based on the idea that with minimum training data the system can learn to drive in traffic on local roads.

The main takeaway from this paper and recommended strategy is :
* Use YUV color space
* Use images from both left and right camera along with center camera
* Use random shift and random rotation as Image Augmentation

![Training Strategy][image8]

The model consists of 9 layers. 1 Lambda Layer ( Normalization Layer ), 5 Convolution Layer, 3 Fully Connected Layer.
My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (code cell 43 of Behavior Cloning Project v2.0.ipynb).
The model includes RELU activation function to introduce nonlinearity (code cell 43 of Behavior Cloning Project v2.0.ipynb), and the data is normalized in the model using a Keras lambda layer (code cell 31 of Behavior Cloning Project v2.0.ipynb).

### Creation of the Training Set & Training Process

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

#### Preprocessing the Data
I preprocessed the data by applying following techniques:

1. I cropped the less useful data, i.e. upper portion with trees and forest, and lower portion with car's hood.
2. Converting the image from RGB color space to YUV color space.
3. Resize the image to 200x66x3

YUV image
![YUV image][image13]

Preprocessed Image
![preprocessed Image][image6]

_Note_ : YUV color space and Image size with 200x66x3 is recommended in the paper . Which gave amazing results and help generalize the model.

### Image Augmentation

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

### Training and Validating the Data

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30.  I used an adam optimizer with learning_rate = 0.0001.

![Loss Graph][image7]

### To run and test the model on simulator

```sh
python drive_updated.py model_v2.0.h5
```
and run the simulator in autonomous mode , choose any track.


### Output Video

Track 1:

output of track 1 is [here](Output_Video/track1_output.mp4)

Track 2

output of track 2 is [here](Output_Video/track2_output.mp4)
