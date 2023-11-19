#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image


# In[ ]:





# In[6]:


from sklearn.model_selection import train_test_split

def get_datasets(dataset):
    # Split the dataset into training, test, and validation sets
    train_data, test_val_data = train_test_split(dataset, test_size=0.30, random_state=42)
    val_data,test_data = train_test_split(test_val_data, test_size=0.15, random_state=42)

    return train_data, test_data, val_data
 


# In[7]:


import os
from sklearn.model_selection import train_test_split

angry_train_folder = 'train/angry'
bored_train_folder="train/bored"
angry_file = os.listdir(angry_train_folder)
bored_file=os.listdir(bored_train_folder)
focused_train_folder = 'train/focused'
neutral_train_folder = 'train/neutral'
focused_file = os.listdir(focused_train_folder)
neutral_file = os.listdir(neutral_train_folder)
angry_train_data, angry_test_data, angry_val_data = get_datasets(angry_file)
bored_train_data, bored_test_data, bored_val_data = get_datasets(bored_file)
focused_train_data, focused_test_data, focused_val_data = get_datasets(focused_file)
neutral_train_data, neutral_test_data, neutral_val_data = get_datasets(neutral_file)


# In[8]:


from PIL import Image
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def get_image_label_pairs(folder_path, label, transform=None):
    image_label_pairs = []
    files = os.listdir(folder_path)
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Filter by image extensions
            image_path = os.path.join(folder_path, file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    # Resize the image to match the expected input size (e.g., 32x32)
                    image = cv2.resize(image, (32, 32))

                    # Convert to float and normalize
                    image = image.astype(np.float32) / 255.0

                    # Ensure the shape includes the channel dimension for PyTorch
                    # Reshape the image to have a single channel (grayscale)
                    image = image.reshape(1, 32, 32)

                    # Convert the NumPy array to a PyTorch tensor
                    image = torch.from_numpy(image)

                    # Apply the specified transformations
                    if transform:
                        image = transform(image)

                    image_label_pairs.append((image, label))
                else:
                    print(f"Skipping {file} due to inability to read the image.")
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
    return image_label_pairs
data_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
])
angry_train_folder = 'train/angry'
bored_train_folder = 'train/bored'
focused_train_folder = 'train/focused'
neutral_train_folder = 'train/neutral'

angry_data = get_image_label_pairs(angry_train_folder, 0, transform=data_transform)
bored_data = get_image_label_pairs(bored_train_folder, 1, transform=data_transform)
focused_data = get_image_label_pairs(focused_train_folder, 2, transform=data_transform)
neutral_data = get_image_label_pairs(neutral_train_folder, 3, transform=data_transform)


angry_train, angry_test, angry_val = get_datasets(angry_data)
bored_train, bored_test, bored_val = get_datasets(bored_data)
focused_train, focused_test, focused_val = get_datasets(focused_data)
neutral_train, neutral_test, neutral_val = get_datasets(neutral_data)

# Combine the splits for each emotion category
combined_training_data = angry_train  + bored_train+ neutral_data+focused_data
combined_test_data = angry_test  + bored_test+neutral_test+focused_test
combined_val_data = angry_val + bored_val+neutral_val+focused_val

batch_size = 32  # Set your desired batch size
custom_training_loader =DataLoader(combined_training_data, batch_size=batch_size, shuffle=True)
custom_testing_loader=DataLoader(combined_test_data, batch_size=batch_size, shuffle=True)
custom_validation_loader=DataLoader(combined_val_data,batch_size=batch_size,shuffle=True)
# Reshape the data to have a single channel (assuming data is grayscale)
print(len(bored_val))


# In[9]:


from collections import defaultdict

# Combined datasets (training, testing, validation)
datasets = [combined_training_data, combined_test_data, combined_val_data]
dataset_names = ['Training', 'Testing', 'Validation']

for dataset, name in zip(datasets, dataset_names):
    label_count = defaultdict(int)
    for _, label in dataset:
        label_count[label] += 1

    print(f"{name} Dataset:")
    for label, count in label_count.items():
        print(f"Label {label}: {count} samples")


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label
custom_training_set = CustomDataset(combined_training_data)
custom_testing_set = CustomDataset(combined_test_data)
custom_validation_set = CustomDataset(combined_val_data)

custom_training_loader = DataLoader(custom_training_set, batch_size=batch_size, shuffle=True)
custom_testing_loader = DataLoader(custom_testing_set, batch_size=batch_size, shuffle=True)
custom_validation_loader = DataLoader(custom_validation_set, batch_size=batch_size, shuffle=True)
    


# In[11]:


class SuperSimpleCNN(nn.Module):
    def __init__(self):
        super(SuperSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Added BatchNorm2d and LeakyReLU to conv1 and conv2
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(32 * 5 * 5, 4)  # Adjusted input size based on the output of the last convolutional layer
        
                

    def forward(self, x):
        x = self.leaky_relu1(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)  # Apply max pooling after the first convolutional layer

        x = self.leaky_relu2(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)  # Apply max pooling after the second convolutional layer
      
        x = x.view(-1, 32 * 5 * 5)  # Adjusted input size for the fully connected layers
        x = self.fc1(x)
        return x


# In[12]:


import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
np.random.seed(seed)
random.seed(seed)


# In[13]:


import torch.optim as optim
def train_model(Model,num_epochs):
    
    model = Model

    # Inside your training loop
    class_weights = torch.tensor([0.5, 0.5, 1.0, 0.8])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
     # Learning rate scheduler

    # Training loop

    # Assuming you have combined_train_loader and validation_loader DataLoader objects

    num_epochs = num_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(custom_training_loader):
            optimizer.zero_grad()
            outputs = model(images)  # Define criterion here with weighted classes
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f"Epoch [{epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Validation after each epoch
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in custom_validation_loader:
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}] Validation Loss: {val_running_loss / len(custom_validation_loader):.3f}")
        print(f"Epoch [{epoch + 1}] Validation Accuracy: {(100 * correct / total):.2f}%")


# In[15]:


SuperSimpleCNN_model=SuperSimpleCNN()
train_model(Model=SuperSimpleCNN_model,num_epochs=45)


# In[18]:



import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
def load_model_and_predict(model, image_path):
    # Add more elif statements for other models as needed
    model.eval()

    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
        transforms.Resize((32, 32)),  # Resize the image to (32, 32)
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
    ])

    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transformation(image)
    with torch.no_grad():
        predictions = model(image.unsqueeze(0))  # Add an extra dimension for batch size

        
    # Print or use predictions as needed
    probabilities = F.softmax(predictions, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    predicted_class = predicted_class.item()

    # Print the results
    print(f"Predicted class: {predicted_class}, Confidence: {confidence.item()}")

    return predicted_class, confidence.item()
    print(predictions.numpy())


# In[20]:


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and run a saved PyTorch model on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    load_model_and_predict(SuperSimpleCNN_model, args.image_path)


# In[ ]:




