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

The only preprocessing technique I used is normalization. The normalization ensures that the pixel values are between 0 and 1. Greyscaling is not used here since my model gives a relatively high prediction accuracy with the pictures having 3 color channels.
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
- epoch = 25. I used 10 epoches initially but the accuracy stuck at around 80%. Thus I used 25 epoches to increase model accuracy.
- batch size = 200
- optimizer = tf.train.AdamOptimizer. Using adam optimizer allows straight forward implementation and efficient computation. Yet adam optimizer is not as geneneralized as SGD.
- learn rate = 0.002. I lowered the learn rate (compared to the tenserflow lab) to improve optimization.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My original accuracy was around 0.8 when I used 10 for epoches and 128 as my batch size. Then I increased the epoches to 25 and kept batch size the same. The accuracy increased, but still below 0.9. Therefore, I changed my batch size to 200 for reaching a desirable prediction accuracy listed below. Techniques to prevent overfitting such as droupout is not applied here, since overfitting is not presented in my testing results.

My final model results were:
* validation set accuracy = 0.932
* test set accuracy of = 0.916

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![][image1] ![][image2] ![][image3] ![][image4] ![][image5]

The images are of 32x32 pixels in dimension. They have relatively good quality, except that in the second image there are a lot of edges in the background. The edges might affect my CNN to recognize the sign. Additionally, the forth image is blurred when zoomed in: the curves that depict children look similar to overlaped squares. This could confuse my CNN too.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction     					| 
|:---------------------:|:---------------------------------------------:| 
| 20km/h (label 0)         		| 35   									| 
| No entry (label 17)    			| 17 										|
| Children crossing (label 28)		| 25											|
| Keep left (label 39) 	   		| 39					 				|
| Curve left (label 19)			| 28      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy is lower than expected. The reason of the mis-prediction could result from extra lines in the background and the low quality of my pictures downloded from the web. Particularly, the "children crossing" image appears to be blurred.

Top five softmax:
- 20km/h (label 0) has predictions: [1 5 0 2 4]; the according probabilities are: [  9.91248131e-01   8.70538969e-03   3.93916853e-05   6.09247854e-06   7.17794990e-07]
- No entry (label 17) has predictions: [17 0 12 29 4]; the according probabilities are: [  9.98516381e-01   1.19396532e-03   1.83336044e-04   4.24175560e-05   3.21090265e-05]
- Children crossing (label 28) has predictions: [25 31 21 30 26]; the according probabilities are: [  9.99511719e-01   4.52846230e-04   3.53927499e-05   2.94016598e-08   2.20060450e-08]
- Keep left (label 39): [39 33 13 35 5]; the according probabilities are: [  9.99958396e-01   3.09650932e-05   8.80030711e-06   1.65821587e-06   9.77689325e-08]
- Curve left (label 19): [28 29 23 30 20]; the according probabilities are: [  9.97260690e-01   2.46191840e-03   2.74204416e-04   2.47562934e-06   2.71457964e-07]


