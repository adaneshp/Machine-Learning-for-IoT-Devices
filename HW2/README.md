# Homework 2
- This code was developed in March, 2022.
- We're going to examine a convolutional neural network (CNN) used for classifying images in the CIFAR-10 dataset. 
    - Each input image is 32x32 pixels with red/green/blue channels for a total size of 32x32x3.
    - The model has two convolutional layers, followed by 2 fully connected layers:
        - Conv 2D: 32 filters, 3x3 kernel, stride=2 (in both x,y dimensions), "same" padding
        - Conv 2D: 64 filters, 3x3 kernel, stride=2 (in both x,y dimensions),  "same" padding
        - MaxPooling, 2x2 pooling size, 2x2 stride
        - Flatten.
        - Dense (aka Fully Connected) , 1024 units.
        - Dense (aka Fully Connected) , 10 units
- Use a spreadsheet or a script to calculate for each layer the number of parameters, the number of multiply-accumulate operations, and the output size.  Turn in a pdf showing a table of parameter and mac counts per layer, as well as the spreadsheet or script you used to calculate it.

- Implement the model in Keras, naming it 'model1', and use the summary() method to check your parameter count and output size calculations.  (Make sure to 'compile' it).
- Load the CIFAR-10 set and split the training set into a training and validation subset.
- Your training, test, and validation sets should be in variables named, respectively: (train_images, train_labels), (test_images, test_labels), (val_images, val_labels)
- Train for 50 epochs.  What is the training, validation, and test accuracy at the end of training? 
    - Did you observe any overfitting?  Should the model train for longer, shorter, or about that number of epochs.
- Take or find a picture of something that is one of the output classes ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'). Save it as a png or jpg, crop and/or rescale it to 32x32 pixels, and run your classifier on it. Name the cropped/scaled image "test_image_classname.ext" where classname is one of the output classes listed and 'ext' is the appropriate filename extension (jpg or png).
    - Does it correctly label the picture?
- Repeat steps 2,4,5 with a model where the convolutional layers are replaced with depthwise-separable convolutions. Name this model 'model2'.

- For this model, did you observe any overfitting?  Should the model train for longer, shorter, or about that number of epochs.
- How did the two models compare in terms of accuracy?
