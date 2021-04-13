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
model_name = "modelTransferPartial.h5"

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

    data = np.append(data, data_i[8000:11000], axis=0)
    target = np.append(target, target_i[8000:11000], axis=0)

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

def binarize(image):
    image[image >= 0.5] = 1
    image[image < 0.5] = 0
    return image

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    # preprocessing_function=binarize,
    fill_mode='nearest')

datagen.fit(x_train)

def transferNoTrainable():
    # Transfer learning Mobilenet model with no layers trainable
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    base_model = MobileNetV2(input_shape=(32,32,3),weights='imagenet',classes=num_classes,include_top=False)
    base_model.summary()

    for layer in base_model.layers:
        if layer.name == 'block_15_expand':
            print('break')
            break
        layer.trainable=False

    last_layer = base_model.get_layer('out_relu')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    input_layer = keras.layers.Input(shape=(28,28,3))
    x = keras.layers.Layer()(input_layer)
    print(x)
    x = layers.experimental.preprocessing.Resizing(32,32,interpolation="bilinear",name="the_resize_layer")(x)
    x = base_model(x)
    # Flatten the output layer to 1 dimension
    x = layers.GlobalAveragePooling2D()(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a final sigmoid layer for classification
    output = layers.Dense(10, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    # model.summary()

    model.compile(optimizer = Adam(lr=0.0001), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])

    model.fit_generator(datagen.flow(x_train, y_train_onehot, batch_size=10),
                        steps_per_epoch=len(x_train)//10,
                        epochs = 10,
                        verbose = 1
    )
    return model

model = transferNoTrainable()

# Save model
model.save(os.path.join(model_dir, model_name + ".h5"))
# Save Tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(os.path.join(model_dir, model_name + '.tflite'), "wb").write(tflite_model)