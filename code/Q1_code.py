import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os
import keras
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import itertools


# All inputs
category = "aeroplane"
true_sample_perc = 50
false_sample_perc = 10


# Download Model
model_vgg16 = keras.applications.VGG16(
    weights="imagenet", input_shape=(224, 224, 3), include_top=False
)
model_vgg16.trainable = False


# !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar -P data/ # noqa
# !tar -xf data/VOCtrainval_25-May-2011.tar -C data/


# Choosing category and placing it in inputs
labels_path = r"data/TrainVal/VOCdevkit/VOC2011/ImageSets/Main"
image_path = r"data/TrainVal/VOCdevkit/VOC2011/JPEGImages"
file_list = os.listdir(labels_path)
for file in file_list:
    if "_train.txt" in file:
        df = pd.read_csv(
            f"{labels_path}/{file}", sep="\s+", names=["filename", "label"]
        )
        true_count = df.loc[df.label == 1].filename.count()
        print(f"{file}->{true_count}")


# Category Chosen for training,
# Sampling data for the chosen category
train_label_path = os.path.join(labels_path, f"{category}_train.txt")
val_label_path = os.path.join(labels_path, f"{category}_val.txt")
train_label_df = pd.read_csv(
    train_label_path, sep="\s+", names=["filename", "label"]
)
test_label_df = pd.read_csv(
    val_label_path, sep="\s+", names=["filename", "label"]
)
train_label_df.loc[train_label_df["label"] != 1, "label"] = 0
test_label_df.loc[test_label_df["label"] != 1, "label"] = 0

# Sampling Data
df_true = train_label_df.loc[train_label_df["label"] == 1].sample(
    frac=true_sample_perc / 100, random_state=22
)
df_false = train_label_df.loc[train_label_df["label"] == 0].sample(
    frac=false_sample_perc / 100, random_state=22
)
# concat both true and false labels and shuffle them to get balanced data
final_train_labels = (
    pd.concat([df_true, df_false], ignore_index=True)
    .sample(frac=1)
    .reset_index(drop=True)
)
test_label_df.groupby("label").count()
final_train_labels.groupby("label").count()


# data pre-processing for model training


# Funtion to images given path
def load(file_path):
    """
    Load and preprocess an image from a file.

    Args:
        file_path (str): The path to the image file.

    Returns:
        tf.Tensor: The preprocessed image.
    """
    # Create a Sequential model to apply a series of transformations to the image # noqa
    transform = keras.models.Sequential(
        [
            keras.layers.experimental.preprocessing.Rescaling(
                1.0 / 255
            ),  # Rescale pixel values to [0, 1]
            # Normalize pixel values
            keras.layers.experimental.preprocessing.Normalization(),
        ]
    )

    # Read the file and decode the image into a tensor
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Apply the transformations to the image
    img = transform(img)

    # Convert the image to float32 data type
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize the image to (224, 224)
    img = tf.image.resize(img, (224, 224))

    return img


# all preprocessing
def data_preprocessing(df: pd.DataFrame):
    images = []
    labels = []
    for row in df.itertuples():
        filename = row.filename
        label = row.label
        image = load(os.path.join(image_path, f"{filename}.jpg"))
        images.append(image)
        labels.append(label)
    labels = np.array(labels).astype("float")
    images = np.array(images)
    return images, labels


X, y = data_preprocessing(final_train_labels)


# Random check
def print_label(cat):
    return (
        f"{category} 'Class A'"
        if cat == 1
        else f"Not {category} 'Class Not A'"
    )


def random_check(X, y):
    image_number = random.randint(0, len(X))
    img = X[image_number]
    cat = y[image_number]
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.title(print_label(cat))
    plt.imshow(img)
    plt.show()


random_check(X, y)


# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
print(X_train.shape)
print(y_train.shape)


# Model intialization
final_model = keras.models.Sequential()
final_model.add(model_vgg16)
# Add a flatten layer
final_model.add(Flatten())
final_model.add(Dense(1, activation="sigmoid"))

# Compile the model with categorical crossentropy loss and adam optimizer
final_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)


# Model Fit
model_fit = final_model.fit(
    X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val)
)


# plotting train and val accuracy in each epoch
accuracy = model_fit.history["accuracy"]
val_accuracy = model_fit.history["val_accuracy"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "y", label="Training Accuracy")
plt.plot(epochs, val_accuracy, "r", label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Reading testing data
X_test, y_test = data_preprocessing(test_label_df)


# Testing random unseen images
image_number = random.randint(0, len(X_test))
test_one = final_model.predict(np.array([X_test[image_number]]))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title(
    f"Test Class : {print_label(y_test[image_number])}\n"
    + f"Predicted Class: {print_label(np.round_(test_one[0]))}"
)
plt.imshow(X_test[image_number])
plt.show()


# Checking Test Matrices
predicted_y = final_model.predict(X_test)
# Convert float values to binary
predicted_y = (np.round_(predicted_y)).reshape(-1)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(y_test, predicted_y)
print(f"Accuracy: {accuracy}")

# Calculate and print precision
precision = metrics.precision_score(y_test, predicted_y)
print(f"Precision: {precision}")

# Calculate and print recall
recall = metrics.recall_score(y_test, predicted_y)
print(f"Recall: {recall}")

# Calculate and print f1-score
f1 = metrics.f1_score(y_test, predicted_y)
print(f"F1-score: {f1}")


# Knn Test
X_flattened = X.reshape(X.shape[0], -1)
X_validation_flattened = X_test.reshape(X_test.shape[0], -1)

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(X_flattened)
normalized_val_features = scaler.transform(X_validation_flattened)


# Train and predict kNN classifiers with different hyperparameters


k_values = [5, 7]
distance_metrics = ["euclidean", "manhattan"]

for k, distance_metric in itertools.product(k_values, distance_metrics):
    # Initialize kNN classifier
    knn_classifier = KNeighborsClassifier(
        n_neighbors=k, metric=distance_metric
    )

    # Train kNN classifier
    knn_classifier.fit(normalized_features, y)

    y_pred = knn_classifier.predict(normalized_val_features)

    accuracy_score_value = metrics.accuracy_score(y_test, y_pred)
    confusion_mat = metrics.confusion_matrix(y_test, y_pred)
    print(f"K Value -> {k}, Distance Metric -> {distance_metric}")
    print("Accuracy:", accuracy_score_value)
    print("Confusion Matrix:\n", confusion_mat)
