#**Behavioural Cloning Project** 

In this project, I built a neural network for behavioural cloning of driving around a test track.
I used the provided simulator to generate a collect data of good driving behaviour based on my own driving.
Then, a convolutional neural network was implemented in Keras to predict the correct steering angle based solely on
image input from the center front camera.
After training an validating the model, I verified that the model is able to successfully drive around the track without leaving the road.

[//]: # (Image References)

[nvidia]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Loss"
[loss]: ./writeup/loss.png "Loss"
[center1]: ./writeup/center_1.jpg "Training image"
[center2]: ./writeup/center_2.jpg "Normal image"
[center2fl]: ./writeup/center_2_fl.jpg "Flipped image"
[left3]: ./writeup/left_3.jpg "Left"
[center3]: ./writeup/center_3.jpg "Center"
[right3]: ./writeup/right_3.jpg "Right"


---
##Files Submitted & Code Quality

My project includes the following files:
* `clone.py` containing the script train the model
* `clone_generator.py` containing the script train the model using a generator
* `src/generator.py` containing a generator function for reading the data while training
* `src/models.py` containing the neural network models used in this project
* `src/read.py` containing the functions to read the entire training data in memory.
* the unmodified `drive.py` for driving the car in autonomous mode
* `model.h5` containing the final trained convolution neural network 
* `writeup_report.md` this file

Please note that my project can also be found on [GitHub](https://github.com/bodetc/CarND-Behavioral-Cloning-P3).

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The `clone.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Similarly, the `clone_generator.py` file contains the code for training the network using a generator. This is useful when the training data does not fit in the memory.

##Model Architecture and Training Strategy

### Solution Design Approach

The overall strategy for deriving a model architecture was to train the model on a limited number of training data (one lap on the first track in each direction).
Then, the model was tested on the first track using the provided `drive.py`.

The first model used was a fully-connected neural network without any hidden layer or activation function (basically a linear regression).
While the performance of the model was extremely poor, it allowed me to quickly test the entire stack: collecting training data, training the model (see `clone.py`), and testing the model in the simulator (with `drive.py`).

The second model used was the LeNet network, already used in the previous project.
The input layers and following convolutional layers were extended to accommodate larger pictures.
The code of the model is in the function `lenet()` of `src/models.py`.

The collected data was augmented by adding flipped images and taking the opposite sign for the steering measurement.
This allows to compensate for the left-turn bias of the first track.
However, while training with the second track, the right-hand driving preference is lost.
In that case, the trained model drives correctly until the bridge, but crashes between the normal track and the dirt road located afterwards.

Then, the data was further augmented by adding side images coming from the left and right cameras.
A correction angle of 0.2 was added or subtracted from the steering angle.
This allows the model to learn to recover from being off-center without having to manually drive to a recovery.

With side camera images, the model manages to make a full lap of the first track. However, it doesn't stay on the main road and takes a shortcut via the dirt road that starts after the bridge.
This model was however already allowing the vehicle to drive around the track without leaving roads or hitting any obstacles.
However, the driving feel a bit erratic as the steering angle oscillates rapidly betwen negative and positive values, and the model tends to overcompensate all of its movements.

In order to go further, the model was changed to the nVidia neural network (see next section and `nvidia()` of `src/models.py`), and the amount of training data was increased by driving two more laps on the first track and one on the second (see below).
The provided training data was also taken into account. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

The final model architecture (In the function `nvidia()` of `src/models.py`) consisted of a convolution neural network
based on the model presented [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and used by nVidia for steering training.
Here is a visualisation of the original architecture.

![alt text][nvidia]

In my implementation, the input image (160x320x3 pixels) was first cropped to 66x320x3, then normalized to values between -0.5 and 0.5.
Then, the normalized input was passed though three convolution layers with 5x5 kernel and depths of 24, 36 and 48.
Each of the convolution layer is followed by a 2x2 Max-Pooling layer.
Then comes two 3x3 convolution layers of depth 64.
The output of the convolution layers is the flattened and followed by three fully connected hidden layers with 100, 50 and 10 nodes.
Finally, the output layer is a fully connected layer with a single output node.

To combat the overfitting, I included dropout layers for all fully connected layers with a dropout rate of 10%.
Keras automatically ignore the dropout layers for validation and evaluation, so that the network doesn't need adaptation for those cases.

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
The first lap was recorded counter-clockwise, and the the second one clockwise.
Here is an example image of center lane driving:

![alt text][center1]

To augment the data set, I also included flipped images and angles to avoid the left turn bias.
Here is an example of an image capture and its flipped counterpart.

![alt text][center2]
![alt text][center2fl]

I did not add captures of the vehicle recovering form the side, as it did not improved the behaviour of the vehicle.
This could be due to a bad technique on my part for recording the data.
Adding side images was doing a great job of keeping the vehicle in the middle of the track, and I deemed it sufficient.
The correction angle for the side image was taken at 0.2. After some manual tuning, this seems to be a good value.
Here is an example of left/center/right images for the same instant:

![alt text][left3]
![alt text][center3]
![alt text][right3]

In order to get more data points, I drove two more laps on the first track, one in each direction.
Then drove one full lap on track two in order to improve the generalisation of the model.
Furthermore, the provided training data was also included.

After the collection process, I had 18864 input lines, amounting to 113184 data points.
I finally randomly shuffled the data set and put 20% of the data into a validation set.
The final number of training and validation data points are 90546 and 22638, respectively. 

With this amount of data, each epoch takes about one hour to run on my machine (a Intel Core i5-4670).
The computer also has a nVidia GPU, but I did not manage to use it yet, as CuDNN seems to crash the graphics driver under Windows.

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
The lowest validation error was achieved at epoch 9, and the trained model of tha epoch was used for testing.
The following plot shows the test and validation loss for each epoch.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][loss]

## Final Testing

### Test run on the second first track

The final test run on first track can be seen in the file `video.mpg`.
The car manages to drive around the track without leaving the drivable portion of the track surface.
It does not not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe.

However, one sees that for some time after passing the bridge, the car follows closely the left side of the road.
I does follow the side of road pretty good though, and doesn't leave the road.
This was not happening before I introduced the training data form the second track, where I was driving in the right lane.
It this case, I was not driving in the center of the road but more following one side of it.
And the model was copying this behaviour on the first track
Due to the flipped images, the model doesn't know the difference between left-hand and right-hand driving.


### Test run on the second track

In the second track (`video_track2.mpg`), the cars successfully follows the right lane of the road until reaching the shadow of the mountain.
From then on, the car simply drives straight ahead until it leave the road and eventually get stuck.
It seems that the model cannot handle properly the change in luminosity of the image.
To combat this, a method similar to the used in the traffic sign project could be used: transformation of the RGB images to YUV and renormalization of the luminance channel.
