#Importing images

#importing libraries
import cv2
import os
import sys

#importing files
import glob

print('Importing images...')
script_dir = os.path.dirname(__file__)
folder_path = os.path.join(script_dir, 'PeopleDS') #letting the computer know this file exists
folder_path = os.path.abspath("/home/nvidia01/my-recognition/PeopleDS") #locating file with images

#loading images (ensure import works)
image_extension = "*.jpg"
image_files = glob.glob(os.path.join(folder_path, image_extension))
for file_path in image_files:
    image = cv2.imread(file_path)
    if image is not None:
        print(f"Loaded: {os.path.basename(file_path)}")
    else:
        print(f"Error: Could not load {os.path.basename(file_path)}")
cv2.destroyAllWindows()


#importing libraries
from PIL import Image
import numpy as np

#convert images to be in a 3D numpy array
image_list = []
target_size = (250, 250) #width x height

for filename in image_files:
    try:
        img = Image.open(filename)
        img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0 #normalize pixel values
        image_list.append(img_array)
    except Exception as e:
        print(f"Error loading or processing {filename}: {e}")

print('Images have been imported')
#print(image_list)

#loading images to ensure import worked
#for path in image_list:
#    try:
#        pics = Image.open(path)
#        pics.show() #this will open each image in a separate viewer
#    except FileNotFoundError:
#        print(f"Error: Image file not found at {path}")
#    except Exception as e:
#        print(f"Error opening image {path}: {e}")

print('Starting to load images... \n(Loading can take around 15 minutes)')
import matplotlib.pyplot as plt
import math

num_images = len(image_list)
cols = 420
rows = math.ceil(num_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3)) #create subplots
axes = axes.flatten

for i, path in enumerate(image_list):
    try:
        pics = Image.open(path)
        axes[i].imshow(pics)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off') #hide axes ticks and labels
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
    except Exception as e:
        print(f"Error opening image {path}: {e}")

#turn off extra subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout() #adjust layout to prevent overlapping
plt.show() #display the plot with all images
print('Finished loading images')