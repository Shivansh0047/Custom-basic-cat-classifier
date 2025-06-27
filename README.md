# Custom Deep Neural Network for Binary Classification of Cats

This is a custom deep neural network built from scratch in NumPy to classify images as "cat" or "non-cat".

## Features

- From-scratch implementation (no high-level frameworks like Keras or PyTorch)
- Multi-layer feed-forward neural network with ReLU and Sigmoid activations
- Trained on the Coursera cat dataset (small images of cats vs. non-cats)
- Supports prediction from uploaded images or webcam capture

## Usage

1. Training the Model

- Change RETRAIN = True in model.py to retrain the model from scratch.
- Then run:

    python model.py

2. Making Predictions

- Choose upload to predict on a local image (must be in the same directory).
- Or choose capture to use your webcam.

Example prompt:

    Do you want to (1) upload an image or (2) capture from webcam? (Enter 1 or 2):

3. Saving and Loading Parameters

- Trained model weights are saved to trained_parameters.npy.
- Reload without retraining by setting RETRAIN = False in model.py.

## Requirements

- Python 3.x
- numpy
- matplotlib
- Pillow
- OpenCV (for webcam capture)

Install with:

    pip install numpy matplotlib pillow opencv-python

## Example Result

    Prediction: "cat"

## License

MIT License
