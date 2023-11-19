#!/usr/bin/env python
# coding: utf-8

# In[148]:


import os
from PIL import Image


# In[149]:


def get_train_file_path(foldername):
    return f"train/{foldername}"


# In[150]:


def validate_file(folder_path):
    image_count=0;
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            image_count=image_count+1
    print(f"{folder_path}")        
    print(f"{image_count}")     


# In[151]:


validate_file(get_train_file_path("angry"))
validate_file(get_train_file_path("bored"))
validate_file(get_train_file_path("focused"))
validate_file(get_train_file_path("neutral"))


# In[152]:


def check_imagesize(folder_path):
    valid=True
    print(f"{folder_path}")
    for filename in os.listdir(folder_path):
        file_path=os.path.join(folder_path,filename)
        if os.path.isfile(file_path):
            size_in_byte=os.path.getsize(file_path)
            if size_in_byte>2500:
                print("Invalid size")
                valid=False
                break        
    return valid


# In[153]:


check_imagesize(get_train_file_path("angry"))
check_imagesize(get_train_file_path("bored"))
check_imagesize(get_train_file_path("focused"))
check_imagesize(get_train_file_path("neutral"))


# In[154]:


def get_test_file_path(folder_path):
    return f"test/{folder_path}"


# In[155]:


validate_file(get_test_file_path("angry"))
validate_file(get_test_file_path("bored"))
validate_file(get_test_file_path("focused"))
validate_file(get_test_file_path("neutral"))


# In[ ]:





# In[ ]:





# In[156]:


import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_sample_images_and_pixel_intensity(folder_path, num_samples=5):
    # Get a list of image file paths in the folder
    image_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg'))]
    
    # Randomly sample 'num_samples' images from the folder
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Plot the images and their pixel intensity distributions
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    for i, image_path in enumerate(sample_images):
        # Display the image
        img = Image.open(image_path)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample Image {i + 1}")
        axes[i, 0].axis('off')
        
        # Display the pixel intensity histogram
        pixel_values = np.array(img.convert('L')).ravel()  # Convert to grayscale and flatten using NumPy
        axes[i, 1].hist(pixel_values, bins=256, range=(0, 256), density=True, color='blue', alpha=0.7)
        axes[i, 1].set_title("Pixel Intensity Histogram")
    
    plt.tight_layout()
    plt.show()


# In[157]:


visualize_sample_images_and_pixel_intensity("train/bored")


# In[158]:


import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def display_random_images_from_class_folder(class_folder, num_images=25, image_size=(128, 128)):
 
    
    # List all image files in the class folder
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Check if there are enough images to sample
    if num_images > len(image_files):
        num_images = len(image_files)

    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)

    num_rows, num_cols = 5, 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i, ax in enumerate(axes.ravel()):
        if i < num_images:
            image_file = selected_images[i]
            image_path = os.path.join(class_folder, image_file)
            img = Image.open(image_path)
            
            # Resize the image to the desired size
            img = img.resize(image_size)
            
            ax.imshow(img)
            ax.axis('off')

    plt.show()

# Example usage:
class_folder = "path_to_class_folder"
display_random_images_from_class_folder("train/bored", 25, (128, 128))


# In[ ]:





# In[159]:


import os
import matplotlib.pyplot as plt

def visualize_class_distribution(data_folder):
    # Get a list of subdirectories (class categories) in the data folder
    class_labels = [label for label in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, label))]
    
    # Count the number of data points in each class
    class_counts = [len(os.listdir(os.path.join(data_folder, label)) ) for label in class_labels]
    
    # Create a bar chart to visualize the class distribution
    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, class_counts, color=['blue', 'red','black','green','yellow'])  # You can customize colors
    plt.xlabel("Classes")
    plt.ylabel("Number of Data Points")
    plt.title("Class Distribution")
    plt.show()


# In[160]:


visualize_class_distribution("train")


# In[ ]:





# In[ ]:





# In[2]:


#####################Part2######################


# In[ ]:





# In[162]:


from sklearn.model_selection import train_test_split

def get_datasets(dataset):
    # Split the dataset into training, test, and validation sets
    train_data, test_val_data = train_test_split(dataset, test_size=0.30, random_state=42)
    val_data,test_data = train_test_split(test_val_data, test_size=0.15, random_state=42)

    return train_data, test_data, val_data


# In[ ]:





# In[163]:


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


# In[164]:


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


# In[165]:


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


# In[166]:


# Assuming custom_training_loader is an instance of DataLoader

for batch in custom_training_loader:
    data, labels = batch  # Assuming it's structured as (data, labels)
    print("Shape of the first batch of data:", data.shape)
    print("Shape of the first batch of labels:", labels.shape)
    print(label)
    # If labels are present
    break  # To inspect just the first batc


# In[167]:


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
    


# In[168]:


def get_confusion_matrix(model):
    
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in custom_testing_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Display or print the confusion matrix
    print("Confusion Matrix:")
    print(cm)


# In[169]:


import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in custom_validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


# In[170]:


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 4)  # Assuming 4 emotion categories
        self.dropout = nn.Dropout(0.5)  # Adding dropout layer

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 4 * 4)  # Flattening before fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.fc2(x)
        return x


# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Adjust 8*8 based on your image size
        self.fc2 = nn.Linear(128, 4)  # 4 classes: angry, bored, focused, neutral

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  # Adjust 8*8 based on your image size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# In[6]:


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


# In[ ]:





# In[ ]:





# In[173]:


import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
np.random.seed(seed)
random.seed(seed)


# In[174]:


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


# In[125]:


SimpleCNN_model=SimpleCNN()
train_model(Model=SimpleCNN_model,num_epochs=50)


# In[126]:


get_confusion_matrix(model=SimpleCNN_model)


# In[128]:


evaluate_model(SimpleCNN_model)


# In[142]:


SuperSimpleCNN_model=SuperSimpleCNN()
train_model(SuperSimpleCNN_model,num_epochs=45)


# In[143]:


get_confusion_matrix(model=SuperSimpleCNN_model)


# In[144]:


evaluate_model(SuperSimpleCNN_model)


# In[145]:


ImprovedCNN_model=ImprovedCNN()
train_model(ImprovedCNN_model,num_epochs=58)


# In[146]:


get_confusion_matrix(model=ImprovedCNN_model)


# In[147]:


evaluate_model(ImprovedCNN_model)


# In[2]:


def load_model_and_predict(model_name, image_path):
    model=model_name()
    # Add more elif statements for other models as needed

    model.load_state_dict(torch.load(f"{model_name}_model.pth"))
    model.eval()

    transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
])

    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transformation(image)

    with torch.no_grad():
        predictions = model(image)

    # Print or use predictions as needed
    print(predictions.numpy())


# In[7]:


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and run a saved PyTorch model on an image.")
    parser.add_argument("model_name", type=str, choices=["SuperSimpleCNN"],
                        help="Name of the model to use")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    load_model_and_predict(args.model_name, args.image_path)


# In[70]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




