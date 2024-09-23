import torch.nn as nn
from torchvision import models

def build_model(arch='resnet18', hidden_units=256, output_size=102):
    # Load appropriate pre-trained model based on architecture

    if arch == 'resnet18':
        model = models.resnet18(weights="IMAGENET1K_V1")
        input_size = 512  # Input size for ResNet18
        classifier_attr = 'fc'  # Classifier attribute for ResNet18
    elif arch == 'vgg13':
        model = models.vgg13(weights="IMAGENET1K_V1")
        input_size = 25088  # Input size for VGG13
        classifier_attr = 'classifier'  # Classifier attribute for VGG13
    else:
        raise ValueError(f"Unsupported architecture {arch}")
        
    # Print the architecture details
    print(f"--- Model details ---")
    print(f"Architecture: {arch}")
    print(f"Input Size: {input_size}")
    print(f"Hidden Layer Size: {hidden_units}")
    print(f"Output Size: {output_size}")
    
    # Define new fully connected (classifier) layer
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_units, output_size),  # 102 flower categories by default
        nn.LogSoftmax(dim=1)
    )

    # Replace model's classifier (fc for ResNet, classifier for VGG)
    setattr(model, classifier_attr, classifier)
    
    return model
