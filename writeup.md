# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./German_traffic_sign/sign1.png "Keep left"
[image2]: ./German_traffic_sign/sign2.png "Dangerous curve left"
[image3]: ./German_traffic_sign/sign3.png "No entry"
[image4]: ./German_traffic_sign/sign4.png "Children crossing"
[image5]: ./German_traffic_sign/sign5.png "Speed limit 20 kph"
[image6]: hist.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Below is a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

![][image6]

The x-axis shows the label on the traffic signs; the y-axis shows the number of occurance.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The images are normalized in the following way. The normalization ensures that the pixel values are between 0 and 1:
- X_train = ((X_train - 127.5) / 127.5)
- X_valid = ((X_valid - 127.5) / 127.5)
- X_test = ((X_test - 127.5) / 127.5)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120  									|
| RELU					| Activation									|
| Fully connected		| outputs 84									|
| RELU					| Activation									|
| Fully connected		| outputs 43 									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
- epoch = 25
- batch size = 200
- optimizer = tf.train.AdamOptimizer
- learn rate = 0.002

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy = 0.929
* test set accuracy of = 0.915

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![][image1] ![][image2] ![][image3] ![][image4] ![][image5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction Correct     					| 
|:---------------------:|:---------------------------------------------:| 
| 20km/h         		| N   									| 
| No entry     			| Y 										|
| Child crossing		| Y											|
| Keep left 	   		| B					 				|
| Curve left			| N      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.




