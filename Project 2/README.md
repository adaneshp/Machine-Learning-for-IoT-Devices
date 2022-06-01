# Project 2

In this project, we aim to design, train, and deploy a model that is able to detect a specific object seen in the Arduino camera (we have chosen motorbike as the target object). This model should run in real-time on Pico4ML board.  As an starting point, we have used the Arduino program for person detection and replaced our Tensorflow model in it. The aim of this project is to learn the challenges and constrains of real-world deployment.

## Training Directory
This folder contains all the Python codes for models that we designed and traiend for detecting motorbike. After finding the best model, it is converted to TensorFlow Lite to be embedded on the board. 

## Embedded Directory
In this folder, the best model from the previous step is embedded to the c/c++ programs to be ready for deployment on the Pico4ML board. Most of these files are provided by the Arduino program for person detection.