# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle


img_rows, img_cols = 28, 28
# Had 3000 but takes way too long to train
num_samples_per_class = 1000
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

    data = np.append(data, data_i[:num_samples_per_class], axis=0)
    target = np.append(target, target_i[:num_samples_per_class], axis=0)

    i += 1
    print("Current data array shape: {}".format(data.shape))
    print("Current target array shape: {}".format(target.shape))


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



# Train a SVM model with rbf kernel using X_train and y_train
rbfK = svm.SVC(kernel='rbf', probability=True, random_state=888)
rbfK.fit(x_train, y_train)
print("Accuracy on test set with rbf kernel SVM model: {:.3f}".format(rbfK.score(x_test, y_test)))

pkl_filename = "svm_model.pkl"
with open('model/svm_model.pkl', 'wb') as file:
    pickle.dump(rbfK, file)

with open('model/svm_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

print("Accuracy on test set with rbf kernel SVM model: {:.3f}".format(pickle_model.score(x_test, y_test)))


# Display confusion matrix
y_pred = pickle_model.predict(x_test)
confusion_matrix = confusion_matrix(y_test.astype(int), y_pred)
print()
print(confusion_matrix)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.xticks(np.arange(0,num_classes), LABELS, rotation=90)
plt.yticks(np.arange(0,num_classes), LABELS)
plt.gcf().subplots_adjust(left=0.05)
plt.gcf().subplots_adjust(bottom=0.40)
plt.show()