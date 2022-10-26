import numpy as np 
import pandas as pd

import os

from PIL import Image
import cv2 
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

Histogram of Gradien
positive_folders = os.listdir("./Positive Sample")
negative_folders = os.listdir("./Negative Sample")

positive_data = np.zeros(len(positive_folders), dtype=object)
for dirname, _, filenames in os.walk("./Positive Sample"):
    for i in range(len(filenames)):
        img = Image.open(os.path.join(dirname, filenames[i]))
        img = img.resize((80,80), Image.ANTIALIAS)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.reshape((80,80,-1))
        positive_data[i]=img

negative_data = np.zeros(len(negative_folders),dtype=object)
for dirname, _, filenames in os.walk("./Negative Sample"):
    for i in range(len(filenames)):
        img = Image.open(os.path.join(dirname, filenames[i]))
        img = img.resize((80,80), Image.ANTIALIAS)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.reshape((80,80,-1))
        negative_data[i]=img

ppc = 8

positive_hog_images = np.zeros(len(positive_data),dtype=object)
positive_hog_features = np.zeros(len(positive_data),dtype=object)

negative_hog_images = np.zeros(len(negative_data),dtype=object)
negative_hog_features = np.zeros(len(negative_data),dtype=object)

for i in range(len(positive_data)):
    fd, hog_image = hog(positive_data[i], orientations=8, pixels_per_cell=(ppc, ppc),
                    cells_per_block=(4, 4), visualize=True)
    positive_hog_images[i] = hog_image
    positive_hog_features[i] = fd

    
for i in range(len(negative_data)):
    fd, hog_image = hog(negative_data[i], orientations=8, pixels_per_cell=(ppc, ppc),
                    cells_per_block=(4, 4), visualize=True)
    negative_hog_images[i] = hog_image
    negative_hog_features[i] = fd

positive_labels = np.array([0 for i in range(len(positive_data))])
negative_labels = np.array([1 for i in range(len(negative_data))])
labels = np.concatenate((positive_labels,negative_labels))

positive_hog_features=positive_hog_features.tolist()
negative_hog_features=negative_hog_features.tolist()
features = [*positive_hog_features, *negative_hog_features]
len(features)
40
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label_encoded=le.fit_transform(labels)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3)

# Function for confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score

def plot_confusion_matrix(y_true, y_pred, classes,
                          positiveize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    positivization can be applied by setting `positive=True`.
    """
    if not title:
        if positiveze:
            title = 'Positiveized confusion matrix'
        else:
            title = 'Confusion matrix, without positiveization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_test, y_pred)]
    if positiveize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.transpose(cm)
        print("Positiveized confusion matrix")
    else:
        print('Confusion matrix, without positiveization')

    print(cm)
    
    
    fig, ax = plt.subplots(figsize = (8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Prediction Label',
           xlabel='Actual Label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=15)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if positiveize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=25,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    ax.grid(False)
    fig.tight_layout()
    return ax

K-Nearest Neighbors Model for K=3
In [11]:
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy for KNN K=3:",metrics.accuracy_score(y_test, y_pred))

Accuracy for KNN K=3: 1.0

In [12]:
from sklearn.metrics import classification_report
print('HOG K-Nearest Neighbors Model for K=3 for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG K-Nearest Neighbors Model for K=3 for YOLO bounding box - Confusion matrix')

HOG K-Nearest Neighbors Model for K=3 for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         8

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

Positiveized confusion matrix
[[1. 0.]
 [0. 1.]]

Out[12]:
<AxesSubplot:title={'center':'HOG K-Nearest Neighbors Model for K=3 for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>
K-Nearest Neighbors Model for K=5
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy for KNN K=5:",metrics.accuracy_score(y_test, y_pred))

Accuracy for KNN K=5: 1.0
from sklearn.metrics import classification_report
print('HOG K-Nearest Neighbors Model for K=5 for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG K-Nearest Neighbors Model for K=5 for YOLO bounding box - Confusion Matrix')

HOG K-Nearest Neighbors Model for K=5 for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         8

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

Positiveized confusion matrix
[[1. 0.]
 [0. 1.]]

<AxesSubplot:title={'center':'HOG K-Nearest Neighbors Model for K=5 for YOLO bounding box - Confusion Matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>

K-Nearest Neighbors Model for K=7
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy for KNN K=7:",metrics.accuracy_score(y_test, y_pred))

Accuracy for KNN K=7: 1.0

from sklearn import metrics
print("Accuracy for KNN K=7:",metrics.accuracy_score(y_test, y_pred))

Accuracy for KNN K=7: 1.0
from sklearn.metrics import classification_report
print('HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         8

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

from sklearn.metrics import classification_report
print('HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Confusion matrix')

HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         8

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

Positiveized confusion matrix
[[1. 0.]
 [0. 1.]]

<AxesSubplot:title={'center':'HOG K-Nearest Neighbors Model for K=7 for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>

Random Forest Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy RF for YOLO bounding box:",metrics.accuracy_score(y_test, y_pred))

Accuracy RF for YOLO bounding box: 0.75
from sklearn.metrics import classification_report
print('HOG Random Forest for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG Random Forest for YOLO bounding box - Confusion matrix')

HOG Random Forest for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       0.60      0.75      0.67         4
           1       0.86      0.75      0.80         8

    accuracy                           0.75        12
   macro avg       0.73      0.75      0.73        12
weighted avg       0.77      0.75      0.76        12

Positiveized confusion matrix
[[0.75 0.25]
 [0.25 0.75]]

<AxesSubplot:title={'center':'HOG Random Forest for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>

Naive Bayes
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy NB for YOLO bounding box:",metrics.accuracy_score(y_test, y_pred))

Accuracy NB for YOLO bounding box: 0.8333333333333334
from sklearn.metrics import classification_report
print('HOG Naive Bayes for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG Naive Bayes for YOLO bounding box - Confusion matrix')

HOG Naive Bayes for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       0.67      1.00      0.80         4
           1       1.00      0.75      0.86         8

    accuracy                           0.83        12
   macro avg       0.83      0.88      0.83        12
weighted avg       0.89      0.83      0.84        12

Positiveized confusion matrix
[[1.   0.25]
 [0.   0.75]]

<AxesSubplot:title={'center':'HOG Naive Bayes for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>
Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy DT for YOLO bounding box:",metrics.accuracy_score(y_test, y_pred))

Accuracy DT for YOLO bounding box: 0.5833333333333334

from sklearn.metrics import classification_report
print('HOG Decision Tree for YOLO bounding box - Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG Decision Tree for YOLO bounding box - Confusion matrix')

HOG Decision Tree for YOLO bounding box - Analysis
              precision    recall  f1-score   support

           0       0.43      0.75      0.55         4
           1       0.80      0.50      0.62         8

    accuracy                           0.58        12
   macro avg       0.61      0.62      0.58        12
weighted avg       0.68      0.58      0.59        12

Positiveized confusion matrix
[[0.75 0.5 ]
 [0.25 0.5 ]]

<AxesSubplot:title={'center':'HOG Decision Tree for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>
Multi Layer Perceptron
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
print("Accuracy MLP for YOLO bounding box:",metrics.accuracy_score(y_test, y_pred))

Accuracy MLP for YOLO bounding box: 0.9166666666666666

from sklearn.metrics import classification_report
print('HOG Multi Layer Perceptron for YOLO bounding box- Analysis')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=np.array(["negative","positive"]), positiveize = True,
                      title='HOG Multi Layer Perceptron for YOLO bounding box - Confusion matrix')

HOG Multi Layer Perceptron for YOLO bounding box- Analysis
              precision    recall  f1-score   support

           0       1.00      0.75      0.86         4
           1       0.89      1.00      0.94         8

    accuracy                           0.92        12
   macro avg       0.94      0.88      0.90        12
weighted avg       0.93      0.92      0.91        12

Positiveized confusion matrix
[[0.75 0.  ]
 [0.25 1.  ]]

<AxesSubplot:title={'center':'HOG Multi Layer Perceptron for YOLO bounding box - Confusion matrix'}, xlabel='Actual Label', ylabel='Prediction Label'>
