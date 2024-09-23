# utils.py
# For data loading, pre-processing image, loading/save check point

import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import build_model 
import os
import matplotlib.pyplot as plt

def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    # Preprocess image
    image_tensor = process_image(image_path)
    
    # Add batch dimension since the model expects a batch
    image_tensor = image_tensor.unsqueeze(0)  # Add batch size dimension

    # Move model and image tensor to the correct device
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Move model to evaluation mode and disable gradient calculations
    model.eval()
    with torch.no_grad():

        output = model(image_tensor)
    
    probabilities = torch.softmax(output, dim=1)  # convert to probabilities
    top_probs, top_indices = probabilities.topk(topk)  # get top k probabilities and indices
    top_probs = top_probs.cpu().numpy().flatten()  # Move to CPU and convert to NumPy
    top_indices = top_indices.cpu().numpy().flatten()  # Move to CPU and convert to NumPy

    # Invert the class_to_idx dictionary to map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Get the class labels corresponding to the top indices
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes
    
    
def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Open the image
    try:
        pil_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
    # Resize image, maintaining aspect ratio: shortest side = 256px
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((256, 256 * pil_image.size[0] // pil_image.size[1]))
    else:
        pil_image.thumbnail((256 * pil_image.size[1] // pil_image.size[0], 256))
    
    # Center-crop image to 224x224
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert image to a NumPy array and scale to [0, 1]
    np_image = np.array(pil_image) / 255.0

    # Normalize  image with mean and std values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimension (Height, Width, Channels) to (Channels, Height, Width)
    np_image = np_image.transpose((2, 0, 1))

    # Convert NumPy array to a tensor
    tensor_image = torch.from_numpy(np_image).float()
    
    return tensor_image


def save_checkpoint(model, train_data, save_dir, arch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save class_to_idx mapping
    model.class_to_idx = train_data.class_to_idx

    # Prepare checkpoint with only necessary data for inference
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'arch': arch
    }
    
    filepath = f"{save_dir}/{arch}_checkpoint.pth"
    
    try:
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    # Retrieve architecture from checkpoint
    arch = checkpoint['arch']
    
    # Rebuild the model architecture using only the arch name
    model = build_model(arch)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# Load data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your data transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    return train_dataloader, valid_dataloader, test_dataloader, train_dataset