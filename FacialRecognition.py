#Facial recognition ml model (grants and denies access to blah blah blah)

print('Beginning imports...')


#imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import imgaug.augmenters as iaa
from PIL import Image
import os
import glob
import cv2
import sys


#importing images
print('Importing images')
script_dir = os.path.dirname(__file__)
folder_path = os.path.join(script_dir, 'PeopleDS') #letting the computer know this file exists
folder_path = os.path.abspath("/home/nvidia01/my-recognition/PeopleDS") #locating file with images


#loading images (ensure import works)
image_extension = "*.jpg"
image_files = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True)
for files in image_files:
    image = cv2.imread(files)
    if image is not None:
        print(f"Loaded: {os.path.basename(files)}")
    else:
        print(f"Error: Could not load {os.path.basename(files)}")
cv2.destroyAllWindows()
print('Images have been imported')


#establishing global variables
image_list = []
label_list = []
target_size = (96, 96)


#get all class names from folder structure
class_names = sorted({os.path.basename(os.path.dirname(f)) for f in image_files})
class_to_label = {name: idx for idx, name in enumerate(class_names)}

for filename in image_files:
    try:
        #open and preprocess image
        img = Image.open(filename)
        img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0 #normalize pixel values
        image_list.append(img_array)
       
        #extract class label from the folder name
        class_name = os.path.basename(os.path.dirname(filename))
        label = class_to_label[class_name]
        label_list.append(label)
       
    except Exception as e:
        print(f"Error loading or processing {filename}: {e}")


#ensuring the labelling worked
print(f"Found classes: {class_names}")
print(f"Total images loaded: {len(image_list)}")
print(f"Total labels loaded: {len(label_list)}")


#split into train, test, and validation
print('Splitting datasets...')

train_images, temp_images, train_labels, temp_labels = train_test_split(
    image_list, label_list, test_size=0.2, stratify=label_list, random_state=42)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)


#preprocessing images
print('Preprocessing images...')

def preprocess_images(image_arrays, target_size=(96, 96)):
    processed_images = []
    for img in image_arrays:
        #if image is not the target size, resize it
        if img.shape[:2] != target_size[::-1]:  #height x width
            img_resized = cv2.resize(img, target_size)
        else:
            img_resized = img
       
        #normalize pixel values to [0, 1] if not done yet
        if img_resized.max() > 1.0:
            img_normalized = img_resized / 255.0
        else:
            img_normalized = img_resized
       
        processed_images.append(img_normalized)
   
    return np.array(processed_images)


#converting lists
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)


#establishing training/testing variables
X_train = preprocess_images(train_images, target_size=(96, 96))
X_val = preprocess_images(val_images, target_size=(96, 96))
X_test = preprocess_images(test_images, target_size=(96, 96))

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)


#checking for imbalanced dataset
print("Train:", np.unique(np.argmax(y_train, axis=1), return_counts=True))
print("Val:", np.unique(np.argmax(y_val, axis=1), return_counts=True))
print("Test:", np.unique(np.argmax(y_test, axis=1), return_counts=True))


#balancing datasets
#compute weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)


#convert to dict
class_weights = dict(enumerate(class_weights))


#getting all class 0 images
class_0_indices = [i for i, label in enumerate(np.argmax(y_train, axis=1)) if label == 0]
class_0_images = X_train[class_0_indices]
class_0_labels = y_train[class_0_indices]


#calculate how many to add to match class 1 count
class_1_count = np.sum(np.argmax(y_train, axis=1) == 1)
class_0_count = len(class_0_images)
num_augments_needed = class_0_count - class_1_count


#set up augmentation plan (in a memory efficient way)
batch_size = 64
augmented_images = []
augmented_labels = []

augmenter = iaa.Sequential([
    iaa.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-10, 10)),
    iaa.Multiply((0.85, 1.15)),
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
    iaa.Resize({"height": 96, "width": 96})
])

print(f"Augmenting class 0 with {num_augments_needed} new samples (batch size: {batch_size})")

i = 0
while len(augmented_images) < num_augments_needed:
    #picking one image
    idx = i % len(class_0_images)
    image = (class_0_images[idx] * 255).astype(np.uint8) #back to uint8
    aug_image = augmenter(images=[image])[0] #augment 1 image
    aug_image = aug_image.astype(np.float32) / 255.0 #normalize again
    #append only one at a time
    augmented_images.append(aug_image)
    augmented_labels.append(class_0_labels[idx])

    i += 1

    #printing progress every 500 images
    if i % 500 == 0:
        print(f"Generated {len(augmented_images)} / {num_augments_needed} augmented images")

augmented_images = np.array(augmented_images, dtype=np.float32)
augmented_labels = np.stack(augmented_labels)

print("X_train shape:", X_train.shape)
print("augmented_images shape:", augmented_images.shape)
print("y_train shape:", y_train.shape)
print("augmented_labels shape:", augmented_labels.shape)

X_train_balanced = np.concatenate((X_train, augmented_images), axis=0)
y_train_balanced = np.concatenate((y_train, augmented_labels), axis=0)

print("Augmentation complete")
print("Balanced training set shape:", X_train_balanced.shape)


#ML model setup
print('Setting up model...')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(2, activation='softmax'))


#checking model architecture
model.summary()


#fitting
print('Training the model...')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_balanced, y_train_balanced,
                    epochs=10,
                    batch_size=8,
                    validation_data=(X_val, y_val))


#plotting accuracy graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.savefig("My_plot.png")

#to see image enter: scp username@your-server-ip:/path/to/My_plot.png .
#username -> your username
#your-server-ip -> IP address
#/path/to/My_plot.png . -> path to folder you want it saved to

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('Test accuracy: ', test_acc)


#identifying the correct face to grant access
print('Identifying faces...')


#ensuring images are in the same order
all_image_files = image_files


#converting image_list to arrays and preprocessing if needed
img_array = preprocess_images(image_list)


#getting the predicted classes from the model
predictions = model.predict(img_array)


#getting predicted class indexes
predicted_classes = np.argmax(predictions, axis=1)


#separating the predicted images that are given and denied access
accessGranted_images = []
accessDenied_images = []

for i, pred in enumerate(predicted_classes):
    if pred == class_to_label['Access Granted']:
        accessGranted_images.append(all_image_files[i])
    elif pred == class_to_label['Access Denied']:
        accessDenied_images.append(all_image_files[i])


#print out the images given and denied access
print('Access granted to: ')
for img in accessGranted_images:
    print(img[39:])