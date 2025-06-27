'''
This model uses a simple Coursera dataset to predict whether a image is a cat or non-cat
It does not uses any framework and all the function are defined below are custom written
You could change the Neural Network Architecture by changing the line 245 layers_dims
The model is designed so all the layers use the Relu activation except the last layer which uses sigmoid activation
'''

import numpy as np
import h5py
from PIL import Image
from dnn_app_utils_v3 import *
import cv2

# Load Coursera cat vs non-cat dataset

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])    
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])    

    classes = np.array(test_dataset["list_classes"][:])          

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Preprocess

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(m_train, -1).T
test_x_flatten = test_x_orig.reshape(m_test, -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

# To know the dimensions of the data , uncomment

'''
print(f"Number of training examples: {m_train}")
print(f"Number of test examples: {m_test}")
print(f"Each image is of size: ({num_px}, {num_px}, 3)")
print(f"train_x shape: {train_x.shape}")
print(f"test_x shape: {test_x.shape}")
'''

# Initialize parameters with He

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# Forward propagation

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z, (A, W, b)

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    else:
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")
    caches.append(cache)
    return AL, caches

# Cost

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    return np.squeeze(cost)

# Backward propagation

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (dZ @ A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = sigmoid_backward(dA, activation_cache)
    return linear_backward(dZ, linear_cache)

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[-1]
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f"dA{l+1}"], current_cache, "relu")
        grads[f"dA{l}"] = dA_prev_temp
        grads[f"dW{l+1}"] = dW_temp
        grads[f"db{l+1}"] = db_temp

    return grads


# Update parameters

def update_parameters(parameters, grads, learning_rate):
    new_parameters = {}
    L = len(parameters) // 2
    for l in range(1, L + 1):
        new_parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
        new_parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]
    return new_parameters

# Model

def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {cost:.4f}")

    return parameters

# To see the evaluation of the model of the dataset, uncomment
'''
print("\nEvaluating on training set...")
_, train_acc = predict(train_x, train_y, parameters)
print(f"Train Accuracy: {train_acc:.2f}%")

print("\nEvaluating on test set...")
_, test_acc = predict(test_x, test_y, parameters)
print(f"Test Accuracy: {test_acc:.2f}%")
'''

# Predict

def predict(X, y, parameters):
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    accuracy = np.mean(predictions == y) * 100
    return predictions, accuracy

# Capture your own image

def capture_image_from_webcam(save_path="webcam_image.jpg"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return None

    print("Press SPACE to capture image, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam - Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            print(f"Image saved to {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return save_path

# Predict your own image

def predict_single_image(filename, parameters, num_px=64, classes=None):
    image = Image.open(filename).convert('RGB').resize((num_px, num_px))
    image = np.array(image) / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    AL, _ = L_model_forward(image, parameters)
    prediction = int((AL > 0.5).item())

    if classes is not None:
        label = classes[prediction].decode("utf-8")
        print(f"\nPrediction: \"{label}\"")
    else:
        if prediction == 1:
            print("\nPrediction: \"cat\"")
        else:
            print("\nPrediction: \"non-cat\"")

# Main control: Train or Load

RETRAIN = False
PARAMS_FILE = "trained_parameters.npy"
layers_dims = [12288, 50, 20, 7, 5, 1]

classes = np.array([b"non-cat", b"cat"])

if RETRAIN:
    print("\n[INFO] Training model from scratch...\n")
    parameters, _ = L_layer_model(
        train_x, train_y, 
        layers_dims, 
        num_iterations=3000, 
        learning_rate=0.01, 
        print_cost=True
    )
    np.save(PARAMS_FILE, parameters)
    print(f"\n[INFO] Model parameters saved to {PARAMS_FILE}")
else:
    print("\n[INFO] Loading model from file...\n")
    parameters = np.load(PARAMS_FILE, allow_pickle=True).item()
    print(f"[INFO] Loaded parameters from {PARAMS_FILE}")

# User Choice: Upload or Capture

print("\n=====================================")
print(" Choose an option for prediction: ")
print("  1. Upload existing image (must be in same directory)")
print("  2. Capture new photo with webcam")
print("=====================================\n")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    print("\n[INFO] You chose to upload an image.")
    filename = input("Enter the filename (must be in the same directory): ").strip()
    predict_single_image(filename, parameters, num_px=64, classes=classes)

elif choice == "2":
    print("\n[INFO] You chose to capture with webcam.")
    webcam_filename = "webcam_image.jpg"
    result_path = capture_image_from_webcam(webcam_filename)
    if result_path:
        predict_single_image(result_path, parameters, num_px=64, classes=classes)
    else:
        print("[INFO] No image captured. Exiting.")

else:
    print("\n[ERROR] Invalid choice. Please run the script again and choose 1 or 2.")