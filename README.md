# HPA-Single-Cell-Classification
Kaggle HPA-Single-Cell-Classification competition solution

### Solution uses...
- HPA's U-net cell segmentator for cell segmentation and localisation from https://github.com/CellProfiling/HPA-Cell-Segmentation
- EfficientNetB2 for classification of individual cells from https://github.com/lukemelas/EfficientNet-PyTorch
- Weakly labelled cell image dataset is downloaded from https://www.kaggle.com/c/hpa-single-cell-image-classification/data
- Various python data science libraries contained in the environment.yml file

To recreate our pyhton environment in anaconda you can run...
`conda env create -f environment.yml`

### Overview
For this competition, solutions are asked to localise and classify the presence of certain organelles across different cells in a microscope image. The training data provided is weakly-labelled, so only images containing many individual cells have an overall image label, while solutions are expected to predict labels for each single cell in images in the unseen test set. Exploration of the dataset and visualisation of the segmentation tool we will be using to localise individual cells is contained in the notebooks folder.

- **notebooks/label_examples.ipynb:**   Exploring label prevelance, correlation, and visualising different rare classes in the dataset.
- **notebooks/segmentation.ipynb:**    Visualising segmentation masks and results produced by HPA's own cell segmentation tool.

## Training a model...

### 1. Segment images to produce training set of single cells with weak labels
We first split the images in the provided training data from images of multiple cells into individual cells using a segmentation model.

segment.py will segment the kaggle data into single cells using the HPA U-net cell segmentation tool. It then puts these images into your specified output file along with a csv describing some metadata around each single cell and its corresponding weak label.

`python src/segment.py --input_dir input/training_data --train_csv input/training_data/train.csv --output_dir input/single_cells`

### 2. Generate training/validation folds
Since this is a multi-label classification problem, we are going to need to use some kind of stratified sampling algorithm to ensure we get a good distribution of labels in training and validation phases. We have a massively imbalanced quantity of different class labels so need to ensure we are training and validating evenly on rare-classes for generalisability and model performance.

To do this we use iterative stratification to decide our training folds (implemented in the wonderful scikit-multilearn package). gen_folds.py evenly splits examples of each label into the training and validation sets and outputs a new csv with a 'fold' column denoting which fold a given cell is assigned to. 

`python src/gen_folds.py --train_csv input/single_cells/train.csv --output_csv input/single_cells/train_folds.csv --n_folds 5`

### 3. Train model
Model training is done with the help of Abhishek Thakur's Tez library. Tez is a lightweight trainer so you have to do most of the work to get a model training yourself, however this means that training is very customisable. As a bonus debugging is ussually simpler as the Tez codebase is a manageable size. 

`python src/train.py --train_csv input/single_cells/train_folds.csv --img_dir input/single_cells --batch_size 32 --num_epochs 5 --fold 1 --save_dir models/`

#### *A Note on resizing of cell images...*
Single cells present in the HPA competition dataset are of varying morphologies, therefore each single cell image is a different size and shape. We resize these to a 260x260 square image as this is the recommended input for the EfficientNetB2 architecture we are performing transfer learning on. While it is true some useful predictive information may be lost in terms of cell morphology when we reshape and slightly warp images to make them fit in our square image format, we think this is a good solution. Using different size cell images and a custom pytorch collate function to accomodate this would have a cost in terms of training speed, and 0-padding images (rather than resizing) up to 260x260 means our network has to spend time learning that 0 padding is to be ignored (Jeremy Howard). Of these 3 options, resizing our single cell images to consistent tiles for training appears to be the simplest and most effective.  

**Our training specification is as follows:**
- Network architecture is an EfficientNetB2 with its final fully connected layer replaced with one which outputs 19 classes. A sigmoid activation function is used as a proxy for class probability. 
- Loss function used is binary cross-entropy. 
- We use an Adam optimiser with cosine annealing learning rate scheduling inspired by recent successful data science competition submissions (https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957). 
- Early stopping is of course used to prevent overfitting on longer training runs.
- For image augmentation, cell images are simply transposed and flipped across the x or y axes with probability 0.5 to aid generalisability when training across multiple epochs. The miscroscope images provided by HPA are highly standardised so exposure and contrast adjustments would not be particularly useful for this problem. Pixel values are normalised to help out our optimiser and are scaled such that inputs are in a similar range to those used in ImageNet corpus.  
