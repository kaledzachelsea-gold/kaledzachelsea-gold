
# Traffic Sign Recognition

## Project Overview

This project uses a convolutional neural network (CNN) to classify German traffic sign images into 43 different categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The project uses Python, OpenCV, NumPy, and TensorFlow. OpenCV is used to load and resize the images, while TensorFlow is used to build and train the neural network.

## Data Processing

The `load_data` function loads images from the 43 category directories. Each image is resized to 30 by 30 pixels so that all images have a consistent input size. The images are also converted from OpenCV's BGR format to RGB format. Each image is assigned a label corresponding to its category directory.

## Neural Network Design and Experimentation

I experimented with a convolutional neural network architecture because convolutional layers are effective at identifying visual patterns in images. The model uses two convolutional layers with 32 and 64 filters respectively. Each convolutional layer uses the ReLU activation function and is followed by a max-pooling layer to reduce the spatial dimensions of the image features.

After the convolutional layers, a flattening layer converts the extracted features into a vector. A dense hidden layer with 128 neurons learns more complex patterns, while a dropout layer with a rate of 0.5 helps reduce overfitting. The final output layer contains 43 neurons and uses softmax to predict the probability of each traffic sign category.

The model uses the Adam optimizer, categorical cross-entropy loss, and accuracy as its evaluation metric. Pixel values are normalized to improve training.

The model is trained for 10 epochs. Its performance is evaluated using a separate test dataset. The test accuracy can be used to assess how well the model recognizes traffic signs it has not seen during training.

## Observations

Convolutional and pooling layers allow the network to learn visual features such as edges, shapes, and patterns. The dense layer combines these features to classify the traffic signs. Dropout is included to help prevent the model from relying too heavily on particular neurons.

The model's final accuracy and loss are determined by running the training and evaluation process on the dataset.
