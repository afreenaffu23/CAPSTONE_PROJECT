import os
import cv2
import numpy as np
import sklearn
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Function to load and preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values between 0 and 1
    return img

# Function to extract features using pre-trained VGG16 model
def extract_features(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    features = base_model.predict(img)
    return features.flatten()

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a custom top layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Set the path to your train dataset directory
train_dataset_dir = "C:/Users/Harthik kp/OneDrive/Desktop/train_img"

# Initialize empty lists for storing images and labels
train_images = []
train_labels = []

# Iterate through the dataset directory
for class_name in os.listdir(train_dataset_dir):
    class_dir = os.path.join(train_dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Iterate through the images in each class directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                # Load and preprocess the image
                img = cv2.imread(image_path)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0  # Normalize pixel values between 0 and 1

                # Append the preprocessed image and corresponding label to the lists
                train_images.append(img)
                if(class_name=='original'):
                    train_labels.append(1)
                else:
                    train_labels.append(0)

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
from sklearn.model_selection import train_test_split

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)

model.fit(train_images, train_labels, epochs= 2, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_images, val_labels)


# Load the trained model weights
model.save('forgery_detection_model.h5')

model.load_weights('forgery_detection_model.h5')

# Image paths for testing
test_image_paths = ["C:/Users/Harthik kp/OneDrive/Desktop/casia/CASIA1/Au/Au_ani_0001.jpg"]
#
# test_image_paths = ['test_image1.jpg', 'test_image]
for image_path in test_image_paths:
    # model.preprocess_image(image_path)
    # Extract features from the test image

    # test_features = extract_features(preprocess_image(image_path))
    # for image_name in os.listdir(class_dir):
    #     image_path = os.path.join(class_dir, image_name)
    #     if image_path.endswith('.jpg') or image_path.endswith('.png'):
    #         # Load and preprocess the image
    #         img = cv2.imread(image_path)
    #         img = cv2.resize(img, (224, 224))
    #         img = img / 255.0  # Normalize pixel values between 0 and 1

    # image_path=preprocess_image(image_path);


    # Predict the probability of forgery
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))

    # Normalize the pixel values
    resized_image = resized_image.astype('float32') / 255.0

    # Add an extra dimension to represent the batch size
    input_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(np.array(input_image))
    probability = prediction[0][0]

    # Print the prediction result
    if probability < 0.5:
        print(f'{image_path} is classified as authentic with probability {probability:.4f}')
    else:
        print(f'{image_path} is classified as forged with probability {probability:.4f}')

import pickle
pickle.dump(model,open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))