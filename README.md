# Autonomous-Cars
This contains all the necessary modules required for programming a self driving car. 
## Behavioral Cloning
Our goal is to teach our Car how to make a lap, using a part of the 1st training track included in the simulation given by Udacity. We want our neural network to drive a bit straight, and 
then make some turns to the right until it reaches the initial point. In principle, to teach 
the neural network, we just need to drive the car, recording images of the road and the 
corresponding steering angle that we applied, a process called behavioral cloning.

Our task is divided into three steps:

• Building the dataset

• Designing and training the neural network

• Integrating the neural network in Carla

We are going to take inspiration from the DAVE-2 system, created by Nvidia.DAVE-2 is a system designed by Nvidia to train a neural network to drive a car, intended 
as a proof of concept to demonstrate that, in principle, a single neural network could be 
able to steer a car on a road. Putting it another way, our network could be trained to drive 
a real car on a real road, if enough data is provided. To give you an idea, Nvidia used 
around 72 hours of video, at 10 frames per second.

