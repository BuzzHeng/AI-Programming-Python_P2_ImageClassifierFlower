# train.py

import argparse
import torch
import torch.optim as optim
from torch import nn
from utils import load_data, load_checkpoint, process_image, save_checkpoint
from model import build_model
import time

# Argument parsing
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a model for flower classification")

    # Positional arguments
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vgg13'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    return parser.parse_args()


# Training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    print("Starting Training...")
    print(f"Device: {device}")
    model.to(device)
    for e in range(epochs):
        epoch_start_time = time.time()
        running_train_loss = 0
        model.train()  # Set model to training mode

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero gradients
            logps = model(inputs)  # Forward pass
            loss = criterion(logps, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize model parameters

            running_train_loss += loss.item()

        # Validation at the end of each epoch
        model.eval()  # Set model to evaluation mode
        running_valid_loss = 0
        accuracy = 0

        with torch.no_grad():  # Disable gradient calculations
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                running_valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Calculate average losses and accuracy
        avg_train_loss = running_train_loss / len(train_loader)
        avg_valid_loss = running_valid_loss / len(valid_loader)
        accuracy /= len(valid_loader)

        epoch_end_time = time.time()
        total_time = epoch_end_time - epoch_start_time
        # Print training and validation results for each epoch
        print(f"Epoch {e+1}/{epochs}.. "
              f"Train loss: {avg_train_loss:.3f}.. "
              f"Validation loss: {avg_valid_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}.. "
              f"Time taken: {int(total_time//60)} minutes {total_time % 60:.2f} seconds")

        model.train()  # Return to training mode for next epoch

    print("Training complete!")


def main():
    args = get_input_args()
    
    start_time = time.time()
    
    # Load data
    print(f"Loading data...")
    train_loader, valid_loader, test_loader, train_data = load_data(args.data_dir)

    # Build model
    print(f"Building model...")
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    
    # Define optimizer based on architecture
    if args.arch.startswith('resnet'):
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    elif args.arch.startswith('vgg'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    # Use GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
 
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    save_checkpoint(model, train_data, args.save_dir, args.arch)

    end_time = time.time()
    total_time = end_time - start_time

    # Print total training time in minutes and seconds
    print(f"Total training time: {int(total_time // 60)} minutes {total_time % 60:.2f} seconds")
    
if __name__ == "__main__":
    main()