# MNIST Digit Classification with Convolutional Neural Networks

This project involves building and training a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Overview

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels. The goal of this project is to accurately classify these digits using a CNN.

## Project Structure

- **Data Loading and Preprocessing**: Load the MNIST dataset, reshape the images to include a single channel, and normalize pixel values.
- **Data Visualization**: Display sample images from the training set using Matplotlib.
- **Model Building**: Create a Sequential model with convolutional, pooling, dropout, and dense layers.
- **Model Compilation**: Compile the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
- **Model Training**: Train the model on the training data.
- **Model Evaluation**: Evaluate the model's performance on the test set.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thisispriyanshugupta/mnist-cnn.git
   cd mnist-cnn
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Load and Preprocess Data:**
   ```python
   import tensorflow as tf
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

   x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
   x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
   x_train = x_train.astype("float32") / 255
   x_test = x_test.astype("float32") / 255
   ```

2. **Visualize Sample Images:**
   ```python
   import matplotlib.pyplot as plt
   %matplotlib inline

   fig, axs = plt.subplots(4, 4, figsize=(20, 20))
   plt.gray()
   for i, ax in enumerate(axs.flat):
       ax.matshow(x_train[i])
       ax.axis("off")
       ax.set_title('Number {}'.format(y_train[i]))
   plt.show()
   ```

3. **Build the Model:**
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

   input_shape = (28, 28, 1)

   model = Sequential([
       Conv2D(28, kernel_size=(3,3), input_shape=input_shape, activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dropout(0.2),
       Dense(10, activation='softmax')
   ])
   ```

4. **Compile the Model:**
   ```python
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   ```

5. **Train the Model:**
   ```python
   model.fit(x_train, y_train, epochs=1)
   ```

6. **Evaluate the Model:**
   ```python
   model.evaluate(x_test, y_test)
   ```

## Results

The initial model achieved an accuracy of approximately 11.35% on the test set, indicating the need for further tuning, additional epochs, or deeper network architectures to improve performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## Acknowledgements

- The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for providing the handwritten digit images.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning frameworks.

---

