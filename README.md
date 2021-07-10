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
around 72 hours of video, at 10 frames per second. We can visualize the model as follows:
![image](https://user-images.githubusercontent.com/64439578/125081076-38330d80-e0e3-11eb-91d2-1654788ffa13.png)

We have defined all the necessary funcyions in helper functions.py and import these functions to main.py whenever required.
The testing code is given by Udacity itself and is as follows:
```
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
#### FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'
 
maxSpeed = 10
 
 
def preProcess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print(f'{steering}, {throttle}, {speed}')
    sendControl(steering, throttle)
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)
 
 
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    ### LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
```
## Lane Detection
The lane detection pipeline follows these steps:

1) Pre-process image using grayscale and gaussian blur
   ```
   def prep_img(img):
   image = mpimg.imread('test_images/solidYellowCurve2.jpg')
   grayscaled = grayscale(image)
   plt.imshow(grayscaled, cmap='gray')
   return cv2.GaussianBlur(grayscaled, (kernel_size, kernel_size), 0)
   ```
2)  Canny Edge Detection
    ```
    def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny
    ```
    ![image](https://user-images.githubusercontent.com/64439578/125083279-ca3c1580-e0e5-11eb-8998-5c3ad588ef0b.png)
3) Create the Region of interest:
   ```
   def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
    (200, height),
    (800, 350),
    (1200, height),]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image
    ```
 4) Hough Transform:
    ```
    def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)
     ```
 5) Display:
    ```
    def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
    ```
To view the output please have a look at detector.mp4 as given in the repository.

## Traffic signs recognition:
### Dataset
German Traffic Sign Recognition Dataset (GTSRB) is an image classification dataset.
The images are photos of traffic signs. The images are classified into 43 classes. The training set contains 39209 labeled images and the test set contains 12630 images. Labels for the test set are not published. To look more into the citation please go to the following link:
https://benchmark.ini.rub.de/gtsdb_news.html.

### Model
The model is designed as follows:
```
cnn_model = Sequential()
cnn_model.add(Conv2D(32,3, 3, input_shape = image_shape, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
cnn_model.add(Dense(output_dim = 43, activation = 'sigmoid'))
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
```
### Model Evaluation
We get a test accuracy of 0.8667
The training validation and loss accuracy is plotted as follows:
![image](https://user-images.githubusercontent.com/64439578/125156641-2190c300-e184-11eb-8f66-50b6b825a0be.png)

![image](https://user-images.githubusercontent.com/64439578/125156686-64eb3180-e184-11eb-9833-52a5f4f69df3.png)
