# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Image Classifier with Deep Learning

## Project Overview

### Part 1 - Developing an Image Classifier

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please refer to the rubric(opens in a new tab) for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize (see here(opens in a new tab) for more information).

This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files on GitHub here(opens in a new tab).

If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter ls -lh to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and re-save the checkpoint.

---

### Part 2 - Building the Command Line Application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.

#### Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

1. **Train**
   
   Train a new network on a data set with train.py

   - **Basic usage:**
     ```
     python train.py data_directory
     ```

   - **Prints out:**
     - Training loss
     - Validation loss
     - Validation accuracy

   - **Options:**
     - Set directory to save checkpoints:
       ```
       python train.py data_dir --save_dir save_directory
       ```
     - Choose architecture:
       ```
       python train.py data_dir --arch "vgg13"
       ```
     - Set hyperparameters:
       ```
       python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
       ```
     - Use GPU for training:
       ```
       python train.py data_dir --gpu
       ```

2. **Predict**
   
   Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.

   - **Basic usage:**
     ```
     python predict.py /path/to/image checkpoint
     ```

   - **Options:**
     - Return top K most likely classes:
       ```
       python predict.py input checkpoint --top_k 3
       ```
     - Use a mapping of categories to real names:
       ```
       python predict.py input checkpoint --category_names cat_to_name.json
       ```
     - Use GPU for inference:
       ```
       python predict.py input checkpoint --gpu
       ```

The best way to get the command line input into the scripts is with the `argparse` module(opens in a new tab) in the standard library. You can also find a nice tutorial for argparse here(opens in a new tab).

---

### Compute and Storage Capacity of Workspaces

#### To Prevent Filling Up Your Workspace:

As you go on to train your models, the models will be saved in your `/home/workspace/saved_models` directory along with the path configuration (`*.pth`) files created in the project directory. Both these "saved_models" directory and ".pth" files are bulky interim files, meaning they will be created every time you attempt to train your ML model. These files with a total size > 2.5Gb can fill up your workspace to the brim, thus causing the workspace-restoration problems when you'd return to the workspace next time.

1. **Delete Large Files:**
   - Delete large interim files and directories before closing the workspace.

2. **Move Large Files:**
   - Move large files (e.g., `.pth`) to the `~/opt` directory, where there is more space. Note that files in this directory are temporary and will not persist after a session ends.

3. **Avoid Storage Issues:**
   - Do not save projects with total file sizes exceeding 2.5GB.

**Caution:** If you neither delete nor move heavy files before closing your workspace, you might face restoration issues, and you will need to contact Udacity support.

---

### Submission Checklist

1. **Jupyter Notebook:** Implement the classifier and save as HTML.
2. **Python Scripts:** Include `train.py` and `predict.py`.
3. **Checkpoint Files:** Ensure their size is manageable (<1GB).
4. **Testing:** Verify that both `train.py` and `predict.py` run successfully.

---

### Rubrics Table

| **Criteria**                       | **Specification**                                                                                       |
|------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Submission Files**               | The submission includes all required files.                                                           |
| **Package Imports**                | All necessary packages and modules are imported in the first cell of the notebook.                   |
| **Training Data Augmentation**     | `torchvision.transforms` are used for random scaling, rotations, mirroring, and/or cropping.         |
| **Data Normalization**             | Training, validation, and testing data are appropriately cropped and normalized.                     |
| **Data Loading**                   | Data for each set (train, validation, test) is loaded using `torchvision.ImageFolder`.               |
| **Data Batching**                  | Data for each set is loaded using `torchvision.DataLoader`.                                           |
| **Pretrained Network**             | A pretrained network (e.g., VGG16) is loaded, and its parameters are frozen.                         |
| **Feedforward Classifier**         | A new feedforward network is defined and trained, using features as input.                          |
| **Training the Network**           | Classifier parameters are trained, while feature network parameters remain static.                   |
| **Validation Loss and Accuracy**   | Validation loss and accuracy are displayed during training.                                          |
| **Testing Accuracy**               | Network accuracy is measured on the test data.                                                       |
| **Saving the Model**               | Trained model is saved as a checkpoint with hyperparameters and `class_to_idx` dictionary.          |
| **Loading Checkpoints**            | A function successfully loads a checkpoint and rebuilds the model.                                   |
| **Image Processing**               | `process_image` function converts a PIL image into input suitable for the trained model.             |
| **Class Prediction**               | `predict` function takes an image and checkpoint, returning top K most probable classes.            |
| **Sanity Checking with Matplotlib**| A figure displays an image with top 5 most probable classes and actual flower names.                 |
| **Training a Network**             | `train.py` successfully trains a network on image datasets.                                          |
| **Training Validation Log**        | Training loss, validation loss, and accuracy are printed as the network trains.                     |
| **Model Architecture**             | `train.py` allows users to choose at least two architectures from `torchvision.models`.             |
| **Model Hyperparameters**          | `train.py` allows users to set hyperparameters (learning rate, hidden units, epochs).               |
| **Training with GPU**              | `train.py` allows users to choose GPU for training.                                                  |
| **Predicting Classes**             | `predict.py` reads an image and checkpoint, printing the most likely class and probability.          |
| **Top K Classes**                  | `predict.py` prints top K classes with probabilities.                                                |
| **Displaying Class Names**         | `predict.py` uses a JSON file to map class values to category names.                                |
| **Predicting with GPU**            | `predict.py` uses GPU for predictions.                                                               |
