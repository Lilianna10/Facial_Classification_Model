
# Facial Classification Model

Using the Jetson Orin, this machine learning (ML) model helps identify faces, and grant or deny access based on an individual's identity.


## Overview

This model is based on the face ID on people’s iPhones. It learns and saves a person’s face using an array of different images and recognizes that face to grant or deny access while keeping track of who it grants access to.


## Outputs

<img width="242" height="524" alt="MLmodelOutputScreenshot" src="https://github.com/user-attachments/assets/43aa5c90-2a90-442a-a54a-1fc85bf802a1" />


###### I simply made the output a print statement of who the model granted access to, but it’s very simple to adjust the code and switch out who would be granted access, or add something that is unlocked with this code.
- Format: label/Name_of_file.jpg

 
## 🤖 Installations 

#### NumPy 


```bash
pip install numpy==1.26.4
```

#### Tensorflow

```bash
pip install tensorflow==2.12.0
```

#### Imgaug

```bash
pip install imgaug==0.4.0
```


## Demo

link to demo



## 🔍 Customizing code 

Import the dataset you want by changing the folder_path
```python
script_dir = os.path.dirname(__file__)
folder_path = os.path.join(script_dir, 'PeopleDS') #letting the computer know this file exists
folder_path = os.path.abspath("/home/nvidia01/my-recognition/PeopleDS") #locating file with images
```

Format: "/home/nvidia01/my-recognition/PeopleDS" = "/path/to/your/file"
- When importing your own dataset, ensure there are named subfolders or the labelling process will fail

Depending on the number of subfolders, number of classes needs to be adjusted when making training, testing, and validation variables and adding softmax layer
```python
train_labels = to_categorical(train_labels, num_classes=2) #line 115
test_labels = to_categorical(test_labels, num_classes=2) #line 116
val_labels = to_categorical(val_labels, num_classes=2) #line 117

model.add(layers.Dense(2, activation='softmax')) #line 220
```

Accuracy graphs can be seen using:
```
scp username@your-server-ip:/path/to/My_plot.png .
```
- username = your SSH username
- your-server-ip = IP address
- /path/to/folder/ = path to folder you want it saved to
- My_plot.png . = name of image it's saved as

Change name of the accuracy graph:
```python
plt.savefig("My_plot.png")
```
- Change "My_plot.png" to the name and image format you want


## ✉️ Feedback 

If you have any feedback, please reach out to me at lilianna.hua@gmail.com

