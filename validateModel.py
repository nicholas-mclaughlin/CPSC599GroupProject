import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import warnings
# warnings.filterwarnings("ignore")

img_rows, img_cols = 28, 28
model_name = "modelFullTransfer.h5"

# Determine data directory
data_dir = os.path.join(sys.path[0], "data")
model_dir = os.path.join(sys.path[0], "model")
print("Data Directory: {}".format(data_dir))

FILES = os.listdir(data_dir)
FILES = [x for x in FILES if x.endswith('.npy')]
FILES = sorted(FILES)
FILES = FILES[0:10]
LABELS = [file.replace(".npy", "") for file in FILES]
num_classes = len(LABELS)

# Dataset with where each entry is flattened numpy image
data = np.array([]).reshape(0, img_rows * img_cols)
target = np.array([]).reshape(0, 1)

i = 0
for numpy_file in FILES:
    print("--> Loading the numpy file: {}".format(numpy_file))
    # Load numpy file
    data_i = np.load(os.path.join(data_dir, numpy_file))
    # Create array of labels based on the class and number of samples in the file
    data_num_samples = len(data_i)
    target_i = np.ones(data_num_samples, dtype=int) * i
    target_i = target_i[:, np.newaxis]

    data = np.append(data, data_i[3000:6000], axis=0)
    target = np.append(target, target_i[3000:6000], axis=0)

    i += 1
    print("Current data array shape: {}".format(data.shape))
    print("Current target array shape: {}".format(target.shape))

print("_________________________")
print("Final data array shape: {}".format(data.shape))
print("Final target array shape: {}".format(target.shape))

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.25, random_state=0)
print("Train Length: {} images".format(len(x_train)))
print("Test Length: {} images".format(len(x_test)))

# for i in range(10):
#     plt.imshow(x_train[i].reshape(28,28))
#     plt.title(LABELS[(int)(y_train[i])])
#     plt.show()

# Normalize image data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Reshape images to 28x28x1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Duplicate channel value so sampel has shape 28x28x3
x_train = np.repeat(x_train,3,3)
x_test = np.repeat(x_test,3,3)

# Switch labels to onehot encoding
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

print("Train Sample Shape: {}".format(x_train.shape))
print("Train Label Shape: {}".format(y_train_onehot.shape))

model = keras.models.load_model(os.path.join(model_dir, model_name))
score = model.evaluate(x_test, y_test_onehot, verbose = 1)
print("\nAccuracy {}".format(score[1]))