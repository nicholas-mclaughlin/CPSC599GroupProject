# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

img_rows, img_cols = 28, 28
num_samples_per_class = 3000
num_cross_validate_per_class = 300
num_classes = 10

cwd = os.getcwd()
data_dir = os.path.join(cwd, "data_reduced")
model_dir = os.path.join(cwd, "model")
print("Data Directory: {}".format(data_dir))
print(data_dir)
 
FILES = os.listdir(data_dir)
FILES = [x for x in FILES if x.endswith('.npy')]
FILES = sorted(FILES)
FILES = FILES[0:10]
 
LABELS = [file.replace(".npy", "") for file in FILES]
# LABELS = [label.replace("full_numpy_bitmap_", "") for label in LABELS]
 
num_classes = len(LABELS)
print(FILES)
print(LABELS)
print(num_classes)

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
 
    data = np.append(data, data_i[:num_cross_validate_per_class], axis=0)
    target = np.append(target, target_i[:num_cross_validate_per_class], axis=0)
 
    i += 1
    print("Current data array shape: {}".format(data.shape))
    print("Current target array shape: {}".format(target.shape))
 
# np.random.shuffle(dataset)
# dataset_len = len(dataset)
 
print("_________________________")
print("Final data array shape: {}".format(data.shape))
print("Final target array shape: {}".format(target.shape))
 
 
 
# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.25, random_state=888)
 
print("Train Length: {} images".format(len(x_train)))
print("Test Length: {} images".format(len(x_test)))
 
# Normalize image data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np.reshape(y_train, (len(y_train)))
y_train = y_train.astype(int)

# SVM Cross Validation
y_train = np.reshape(y_train, (len(y_train)))
y_train = y_train.astype(int)
linearK = svm.SVC(kernel='linear', random_state=888)
scores1 = cross_val_score(linearK, x_train, y_train, cv=5)
print("Linear SVC has %0.3f accuracy with a standard deviation of %0.3f" % 
     (scores1.mean(), scores1.std()))

polyK = svm.SVC(kernel='poly', random_state=888)
scores2 = cross_val_score(polyK, x_train, y_train, cv=5)
print("Poly SVC has %0.3f accuracy with a standard deviation of %0.3f" % 
      (scores2.mean(), scores2.std()))

rbfK = svm.SVC(kernel='rbf', random_state=888)
scores3 = cross_val_score(rbfK, x_train, y_train, cv=5)
print("Rbf SVC has %0.3f accuracy with a standard deviation of %0.3f" % 
      (scores3.mean(), scores3.std()))

sigK = svm.SVC(kernel='sigmoid', random_state=888)
scores4 = cross_val_score(sigK, x_train, y_train, cv=5)
print("Sigmoid SVC has %0.3f accuracy with a standard deviation of %0.3f" % 
      (scores4.mean(), scores4.std()))