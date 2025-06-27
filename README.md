# Custom Deep Neural Network for Binary Classification of Cats

This project implements a custom deep neural network from scratch (using only NumPy) to classify images as "cat" or "non-cat".

It demonstrates how to build all layers, activations, forward and backward passes without any high-level frameworks.

---

## Features

- From-scratch implementation (no Keras / TensorFlow / PyTorch)
- Multi-layer feed-forward neural network
- ReLU activations for hidden layers, Sigmoid for output layer
- Predicts using local images or directly from webcam
- Saves and loads trained model weights

---

## Note on Model Limitations

This model is very basic and is not able to distinguish between different cat species (for example, domestic cats vs. lions or tigers). It also misclassifies a lot of images.

It could be made better by training on a bigger and more diverse dataset, but the point of this project is to learn how to implement a deep neural network **from scratch**.

---

## Dataset

This model uses a simple Coursera cat dataset to predict whether an image is a cat or non-cat.
Note: It's a small dataset meant for educational purposes.

---

## How It Works

- All neural network functions (initialization, forward/backward propagation, updates) are custom written in pure NumPy.
- You can change the Neural Network architecture by editing:

  layers_dims = [ ... ]

  For example (line 245 in model.py):

  layers_dims = [12288, 50, 20, 7, 5, 1]

- The model is designed so all the layers use the ReLU activation except the last layer which uses the Sigmoid activation.

---

## Usage

1. Training the Model

- To train from scratch, open model.py and set:

  RETRAIN = True

- You can also change the learning rate and number of iterations in the call to:

  L_layer_model(..., learning_rate=0.01, num_iterations=3000)

- Then run:

  python model.py

- After training, the model weights are saved to trained_parameters.npy.

---

2. Making Predictions

- When you run:

  python model.py

- You will be prompted:

  Do you want to (1) upload an image or (2) capture from webcam? (Enter 1 or 2):

- Option 1: Upload

  Enter 1 and then provide the filename when prompted.
  The image must be in the same directory as model.py or provide the correct relative path.

- Option 2: Capture from Webcam

  Enter 2 to activate your webcam and take a live photo for prediction.

---

3. Saving and Loading Parameters

- Trained model weights are saved in trained_parameters.npy.
- To load an existing model without retraining, set:

  RETRAIN = False

---

## Requirements

- Python 3.x
- numpy
- matplotlib
- Pillow
- OpenCV (for webcam capture)

Install all with:

  pip install numpy matplotlib pillow opencv-python

---

## Example Result

  Prediction: "cat"

---

## Pushing to GitHub

After editing and committing your changes locally, you can push them to your GitHub repository with:

  git push origin main

---

## License

MIT License
