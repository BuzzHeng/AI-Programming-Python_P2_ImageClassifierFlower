import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import process_image, load_checkpoint, imshow, predict
import os

def load_category_names(category_names_path):
    """ Load a JSON file containing category-to-name mapping """
    with open(category_names_path, 'r') as f:
        return json.load(f)

def get_input_args():
    """ Parse command line arguments using argparse """
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with class probability")
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def display_prediction(image_path, probs, classes, cat_to_name):
    """ Display an image along with the top K predicted classes and their probabilities """
    # Map predicted class indices to flower names
    flower_names = [cat_to_name.get(str(cls), "Unknown") for cls in classes]
    print(f"Predicted Classes (name):\n{flower_names}")
    
    # Display image and predictions
    plt.figure(figsize=(6,10))

    # Subplot 1: Display image
    ax = plt.subplot(2, 1, 1)
    imshow(process_image(image_path), ax=ax)
    ax.set_title(flower_names[0])  # Set the title as the most likely class

    # Subplot 2: Display bar chart
    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(flower_names))
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, flower_names)
    plt.gca().invert_yaxis()  # Flip the bar chart for better readability
    plt.xlabel('Probability')
    plt.show()

def main():
    args = get_input_args()

    # Load the model from the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Set the device to GPU if available and requested, otherwise CPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the category-to-name mapping
    cat_to_name = load_category_names(args.category_names)
    
    # Predict the top classes and probabilities
    probs, classes_id = predict(args.image_path, model, topk=args.top_k, device=device)

    # Display the results
    print(f"Predicted Probabilities:\n{probs}")
    print(f"Predicted Classes (IDs):\n{classes_id}")
    display_prediction(args.image_path, probs, classes_id, cat_to_name)

if __name__ == '__main__':
    main()
