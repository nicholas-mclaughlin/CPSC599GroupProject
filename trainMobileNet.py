import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import warnings
# warnings.filterwarnings("ignore")

# Setup variables
img_rows, img_cols = 28, 28
model_name = "modelMobilenet"
num_samples_per_class = 3000
num_classes = 10
epochs = 10

# Determine data directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, "data_reduced")
model_dir = os.path.join(cwd, "model")
print("Data Directory: {}".format(data_dir))

# Look at the data directory files, get a list of files in the directory,
# reduce to the first 10 files, parse the labels from the file names
FILES = os.listdir(data_dir)
FILES = [x for x in FILES if x.endswith('.npy')]
FILES = sorted(FILES)
FILES = FILES[0:num_classes]
LABELS = [file.replace(".npy", "") for file in FILES]


# Create empty dataset with where each entry is flattened numpy image
data = np.array([]).reshape(0, img_rows * img_cols)
target = np.array([]).reshape(0, 1)

# Loop through loading numpy files and set up dataset and target arrays
i = 0
for numpy_file in FILES:
    print("--> Loading the numpy file: {}".format(numpy_file))
    # Load numpy file
    data_i = np.load(os.path.join(data_dir, numpy_file))
    # Create array of labels based on the class and number of samples in the file
    data_num_samples = len(data_i)
    target_i = np.ones(data_num_samples, dtype=int) * i
    target_i = target_i[:, np.newaxis]

    # Append current data to combined array
    data = np.append(data, data_i[:num_samples_per_class], axis=0)
    target = np.append(target, target_i[:num_samples_per_class], axis=0)

    i += 1
    print("Current data array shape: {}".format(data.shape))
    print("Current target array shape: {}".format(target.shape))

print("_________________________")
print("Final data array shape: {}".format(data.shape))
print("Final target array shape: {}".format(target.shape))

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.25, random_state=0)
print("_________________________")
print("Train Length: {} images".format(len(x_train)))
print("Test Length: {} images".format(len(x_test)))

# Normalize image data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Reshape images to 28x28x1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Duplicate channel value so sample has shape 28x28x3
x_train = np.repeat(x_train,3,3)
x_test = np.repeat(x_test,3,3)

# Switch labels to onehot encoding
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

print("_________________________")
print("Train Sample Shape: {}".format(x_train.shape))
print("Train Label Shape: {}".format(y_train_onehot.shape))
print("_________________________")

# Define image augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    fill_mode='nearest')

datagen.fit(x_train)

def trainMobilenetFullyTrainable():
    # Transfer learning Mobilenet model with no layers trainable
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    # Base model is a MobileNet without pretrained weights
    base_model = MobileNetV2(input_shape=(32,32,3),weights=None,classes=num_classes,include_top=True)

    # Define model using functional method, input -> resize -> MobileNet
    input_layer = keras.layers.Input(shape=(28,28,3))
    x = keras.layers.Layer()(input_layer)
    x = layers.experimental.preprocessing.Resizing(32,32,interpolation="bilinear",name="the_resize_layer")(x)
    output = base_model(x)
    model = Model(inputs=input_layer, outputs=output)

    # Compile model
    model.compile(optimizer = Adam(lr=0.0001), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])

    # Train model with samples flowing from ImageDataGenerator
    model.fit_generator(datagen.flow(x_train, y_train_onehot, batch_size=10),
                        steps_per_epoch=len(x_train)//10,
                        epochs = epochs,
                        verbose = 1
    )
    return model

# Call training function
model = trainMobilenetFullyTrainable()

# Validate on test dataset
score = model.evaluate(x_test, y_test_onehot, verbose = 1)
print("\nAccuracy on Test Set: {}".format(score[1]))

# Save model
model.save(os.path.join(model_dir, model_name + ".h5"))

# Display confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = np.argmax(model.predict(x_test, verbose=1), axis=1)
confusion_matrix = confusion_matrix(y_test.astype(int), y_pred)
print()
print(confusion_matrix)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.xticks(np.arange(0,num_classes), LABELS, rotation=90)
plt.yticks(np.arange(0,num_classes), LABELS)
plt.gcf().subplots_adjust(left=0.05)
plt.gcf().subplots_adjust(bottom=0.40)
plt.show()