# HPA-Single-Cell-Classification
Kaggle HPA-Single-Cell-Classification competition solution

### Solution uses...
- HPA's U-net cell segmentator for cell segmentation and localisation
- Transfer learned EfficientNetB2 for classification of individual cells
- Weakly labelled cell image dataset is downloaded from https://www.kaggle.com/c/hpa-single-cell-image-classification/data

### Overview
For this competition, solutions are asked to localise and classify the presence of certain organelles across different cells in a microscope image. Training data provided is weakly-labelled, so only images containing many individual cells have an overall image label, while solutions are expected to predict labels for each single cell in images in the unseen test set. Exploration of the dataset and visualisation of the segmentation tool we will be using to localise individual cells is contained in the notebooks folder.

- notebooks/label_examples.ipynb   Exploring label prevelance, correlation, and visualising different rare classes in the dataset.
- notebooks/segmentation.ipynb    Visualising segmentation masks and results produced by HPA's own cell segmentation tool.

## Training a model...

### 1. Segment images to produce training set of single cells
We need to first split the images in the provided training data from images of groups of cells into individual cells using a segmentation model.

segment.py will segment the provided training data into single cells using the HPA U-net cell segmentation tool and puts these images into your specified output file along with a csv describing some metadata around each single cell and its corresponding weak label.

`python src/segment.py --input_dir input/training_data --train_csv input/training_data/train.csv --output_dir input/single_cells`

### 2. Generate training/validation folds
Since this is a multi-label classification problem, we are going to need to use some kind of stratified sampling to ensure we get a good distribution of labels in training and validation phases. We also have a massively imbalanced quantity of different class labels so need to ensure we are training on rare-classes again for generalisability and model performance.

To do this we use iterative stratification to decide our training folds implemented in the wonderful scikit-multilearn package. gen_folds.py evenly splits examples of each label in the training and validation sets across n_folds and outputs a new csv with fold column denoting which fold a given cell is assigned to. 

`python src/gen_folds.py --train_csv input/single_cells/train.csv --output_csv input/single_cells/train_folds.csv --n_folds 5`

### 3. Train model
Model training is done using Abhishek Thakur's Tez library. Tez is a lightweight trainer, so you have to do most of the work to get a model training yourself, however this is good for us as it means the training is very customisable, and things are easy to debug as the Tez codebase is so small. 

`python src/train.py --train_csv input/single_cells/train_folds.csv --img_dir input/single_cells --batch_size 32 --num_epochs 5 --fold 1 --save_dir models/`

Our training specification is as follows:
- The effnet is trained on single cells. Single cells are of different morphology and therefore each single cell image is a different size. We resize to a 260x260 square image as this is the recommended input for the EfficientNetB2 we are transfer learning on. We are aware some information may be lost in terms of cell morphology when we reshape and slightly warp images to make them fit in our square image format, however we think this is a good trade off. Especially since using different size cell images and a custom collate function would have a cost in terms of training speed, and 0-padding images up to 260x260 means the network has to learn that 0 padding is to be ignored (Jeremy Howard). It was not worth it to do this any other way in our estimation. Single cells are also transposed, and flipped across the x and y axes with probability 0.5 to aid generalisability when training across multiple epochs. Pixel values are normalised to help out our optimiser and are scaled such that inputs are similar to those used in imagenet which the effnet is pretrained on. 
- Our model is an efficientnetb2 with its final fully connected layer replaced to one which outputs 19 classes with a sigmoid activation function as a proxy for class probability. We use a binary cross entropy loss function, and Adam optimiser. We use cosine annealing learning rate scheduling inspired by recent competition competitors. This has become a very popular choice in the competitive data science space, and generally allows for fairly speedy training (https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957). Early stopping is used to prevent overfitting on longer training runs. 
