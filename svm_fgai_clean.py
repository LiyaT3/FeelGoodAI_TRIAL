# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#LiyaT
"""
import os
from PIL import Image

# Function to delete excess images in a folder
def delete_excess_images(folder_path, max_images=3000):
    image_extensions = ('.png', '.jpg', '.jpeg')

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    if len(images) > max_images:
        images_to_delete = len(images) - max_images
        images.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

        for i in range(images_to_delete):
            image_path = os.path.join(folder_path, images[i])
            os.remove(image_path)
            print(f"Deleted: {image_path}")

# Function to visit each folder and delete excess images
def delete_images_in_folders(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            delete_excess_images(folder_path)

# Provide the path to the parent directory (mydata folder)
parent_directory = 'C:\\Users\\ADMIN\\Desktop\\mydata'

delete_images_in_folders(parent_directory)"""






















#IMPORTING LIB
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from PIL import Image
import keras
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn import svm
from tqdm.notebook import tqdm
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
import cv2

train_dir = 'C:\\Users\\ADMIN\\Desktop\\mydata'
test_dir = 'C:\\Users\\ADMIN\\Desktop\\testdata'


def load_data(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                image_paths.append(image_path)
                labels.append(label)
                
            print(label, "Done!")

    return image_paths, labels

train = pd.DataFrame()
train['image'] , train['label'] = load_data(train_dir)

## Shuffling the dataset
train = train.sample(frac =1).reset_index(drop=True) # To shuffle the data for randomly distributing data into dataset
train.head()


test = pd.DataFrame()
test['image'] , test['label'] = load_data(test_dir)
test.head()


#EDA

def EDA(name):
    sns.countplot(data = name , x='label') #Plotting label column of train

    plt.xlabel('Categories - Emotions')
    plt.ylabel('Count')
    plt.title('Count Plot of Emotions Categories')

    plt.show()  
EDA(test)
EDA(train)

# # Open an image file
# img = Image.open(train['image'][1])
# plt.imshow(img , cmap='gray');

""" 
from PIL import Image
plt.figure(figsize=(25,25))
files=train.iloc[:10]
for index , file , label in files.itertuples():
    plt.subplot(5,5,index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.axis('off')
    plt.title(label)"""
    
def extract_feature(images):
    feature = []
    for image in tqdm(images):
        img = load_img(image , color_mode = "grayscale")
        img = np.array(img)
        feature.append(img)
    feature = np.array(feature)
    feature = feature.reshape(len(feature),48,48,1)
    return feature
    
    
train_feature = extract_feature(train['image'])       
test_feature = extract_feature(test['image'])        


x_train = train_feature/255.0
x_test = test_feature/255.0            
        
"""
def encoder(labels):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    
    print("Label to Integer Mapping:")
    for label, integer_label in zip(labels, integer_labels):
        print(f"{label} -> {integer_label}")

# Define your list of labels
labels = ['angry', 'sad', 'fearful', 'happy', 'disgusted', 'surprise', 'neutral']

# Call the encoder function to create the mapping
encoder(labels)

    labels = ['angry', 'sad', 'fearful', 'happy', 'disgusted','surprise','neu
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    print("Label to Integer Mapping:")
    for label, integer_label in zip(labels, integer_labels):
        print(f"{label} -> {integer_label}")

encoder(train['label'])       

lab = LabelEncoder()
lab.fit(train['label'])    

y_train = lab.transform(train['label'])
y_test = lab.transform(test['label'])    


print(y_train)    
print(y_test)    

# # Reshape x_train if needed (e.g., flattening for images)
x_train = x_train.reshape(x_train.shape[0], -1)
# # Ensure y_train is a 1D array
y_train = y_train.ravel()"""

"""
def encoder(labels):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    
    print("Label to Integer Mapping:")
    for label, integer_label in zip(labels, integer_labels):
        print(f"{label} -> {integer_label}")"""


#ENCODING
lab_enc = LabelEncoder()
lab_enc.fit(train['label'])
y_train = lab_enc.transform(train['label'])
y_test = lab_enc.transform(test['label'])

#RESHAPE IT

x_train_2d = x_train.reshape(x_train.shape[0], -1)
x_test_2d = x_test.reshape(x_test.shape[0], -1)

#MODEL BUILDING
svm_model = svm.SVC(kernel='linear')
svm_model.fit(x_train_2d, y_train)

# EVALUATION
accuracy = svm_model.score(x_test_2d, y_test)

print("Accuracy: {}%".format(accuracy*100))


#radial basis func
svm_model1 = svm.SVC(kernel='rbf')
svm_model1.fit(x_train_2d, y_train)

# Evaluate the model on the test data
accuracy1 = svm_model1.score(x_test_2d, y_test)

print("Accuracy: {}%".format(accuracy1 * 100))


#poly
svm_model2 = svm.SVC(kernel='poly')
svm_model2.fit(x_train_2d, y_train)

# Evaluate the model on the test data
accuracy2 = svm_model2.score(x_test_2d, y_test)

print("Accuracy: {}%".format(accuracy2 * 100))



















