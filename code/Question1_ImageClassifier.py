import tensorflow as tf
import os
import random
import numpy as np
from shutil import copyfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

dataset_path = "C:\M.Tech.Stuff\DeepLearning\Assignment\First\VOCtrainval\TrainVal\VOCdevkit\\"
train_path = os.path.join(dataset_path, 'VOC2011\\ImageSets\\Main\\aeroplane_train.txt')
validation_path = os.path.join(dataset_path, 'VOC2011\\ImageSets\\Main\\aeroplane_val.txt')
output_path_value = 'output_data'

os.makedirs(os.path.join(output_path_value, 'A'), exist_ok=True)
os.makedirs(os.path.join(output_path_value, 'not_A'), exist_ok=True)

with open(train_path, 'r') as f:
    train_files = f.read().splitlines()

with open(validation_path, 'r') as f:
    validation_files = f.read().splitlines()

#############################################################################################################################################
#Training dataset representation
#############################################################################################################################################
category_A_images = []
not_category_A_images = []

for row in train_files:
    fileName, label = row.split()
    if label == "1":
        category_A_images.append(fileName)
    else:
        not_category_A_images.append(fileName)

random.shuffle(category_A_images)
random.shuffle(not_category_A_images)

# Randomly select 20% images from category A and 10% from not category A
num_category_A = int(0.2 * len(category_A_images))  # You can adjust the percentage
num_not_category_A = int(0.1 * len(not_category_A_images))  # 10% from not category A

for filename in category_A_images[:num_category_A]:
    source_path = os.path.join(dataset_path, 'VOC2011\\JPEGImages', filename + '.jpg')
    dest_path = os.path.join(output_path_value, 'A', filename + '.jpg')
    copyfile(source_path, dest_path)

for filename in not_category_A_images[:num_not_category_A]:
    source_path = os.path.join(dataset_path, 'VOC2011\\JPEGImages', filename + '.jpg')
    dest_path = os.path.join(output_path_value, 'not_A', filename + '.jpg')
    copyfile(source_path, dest_path)

# extracted features
features = []
labels = []

# Loop through images and extract features
for folder in ['A', 'not_A']:
    image_folder = os.path.join(output_path_value, folder)
    label = 1 if folder == 'A' else 0
    count=0
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Extract features from the last fully-connected layer
        features.append(model.predict(img))
        labels.append(label)

        count=count + 1
        if count == 3000:
            break

# Convert features and labels to numpy arrays
features = np.vstack(features)
labels = np.array(labels)

#############################################################################################################################################
#Validation dataset representation
#############################################################################################################################################
val_category_A_images = []
val_not_category_A_images = []

for row in validation_files:
    fileName, label = row.split()
    if label == "1":
        val_category_A_images.append(fileName)
    else:
        val_not_category_A_images.append(fileName)

# Randomly shuffle and select the desired percentage of images
random.shuffle(val_category_A_images)
random.shuffle(val_not_category_A_images)

# Randomly select 20% images from category A and 10% from not category A
val_num_category_A = int(0.2 * len(val_category_A_images))  # You can adjust the percentage
val_num_not_category_A = int(0.1 * len(val_not_category_A_images))  # 10% from not category A

for filename in val_category_A_images[:val_num_category_A]:
    source_path = os.path.join(dataset_path, 'VOC2011\\JPEGImages', filename + '.jpg')
    dest_path = os.path.join(output_path_value, 'A', filename + '.jpg')
    copyfile(source_path, dest_path)

for filename in val_not_category_A_images[:val_num_not_category_A]:
    source_path = os.path.join(dataset_path, 'VOC2011\\JPEGImages', filename + '.jpg')
    dest_path = os.path.join(output_path_value, 'not_A', filename + '.jpg')
    copyfile(source_path, dest_path)

val_features = []
val_labels = []

for folder in ['A', 'not_A']:
    image_folder = os.path.join(output_path_value, folder)
    label = 1 if folder == 'A' else 0
    count=0
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Extract features from the last fully-connected layer
        val_features.append(model.predict(img))
        val_labels.append(label)

        count=count + 1
        if count == 3000:
            break

val_features = np.vstack(val_features)
val_labels = np.array(val_labels)

X_train_flattened = features.reshape(features.shape[0], -1)
X_validation_flattened = val_features.reshape(val_features.shape[0], -1)

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(X_train_flattened)
normalized_val_features = scaler.transform(X_validation_flattened)

# Train and predict kNN classifiers with different hyperparameters
k_values = [1, 3, 5, 7]
distance_metrics = ['euclidean', 'manhattan']

for k in k_values:
    for distance_metric in distance_metrics:
        # Initialize kNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        
        # Train kNN classifier
        knn_classifier.fit(normalized_features, labels)

        y_pred = knn_classifier.predict(normalized_val_features)

        accuracy_score_value = accuracy_score(val_labels, y_pred)
        confusion_mat = confusion_matrix(val_labels, y_pred)

        print("Accuracy:", accuracy_score_value)
        print("Confusion Matrix:\n", confusion_mat)