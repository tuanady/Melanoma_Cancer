# Melanoma Cancer Detection

This repository contains code for a melanoma cancer detection project using deep learning techniques. The project aims to develop a model capable of accurately classifying skin lesion images as either benign or malignant melanomas.

## Setup and Dependencies

The code is written in Python and relies on several libraries and frameworks. The main dependencies include:

- **NumPy**: For efficient numerical computations and array manipulation.
- **Seaborn** and **Matplotlib**: For data visualization and plotting.
- **Pandas**: For data manipulation and organization.
- **scikit-learn (sklearn)**: For machine learning utilities and metrics.
- **TensorFlow** and **Keras**: For building and training deep learning models.
- **OpenCV (cv2)**: For image processing tasks.
  
## Running the code 

To make this code run, follow the steps

1. git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
2. git -C ~/.pyenv pull
3. exec "$SHELL"
4. pyenv commands
5. export PATH="$HOME/.pyenv/bin:$PATH"
6. eval "$(pyenv init --path)"
7. eval "$(pyenv virtualenv-init -)"
8. pyenv virtualenv 3.11.2 myenv
9. pyenv activate myenv
10. pip install kernel 
11. pip install opencv

## Usage

### Data Preparation

The dataset consists of skin lesion images, which are loaded and prepared for training and testing. Data augmentation techniques are applied to increase the diversity of the training set.

### Model Definition

Two different model used for this research. One of them is custom CNN and the other one pre-trained ResNet50 model.

### Model Training

The model is compiled with appropriate loss and optimization functions, then trained on the training data. Early stopping is implemented to prevent overfitting.

### Model Evaluation

The trained model is evaluated on the testing dataset using various metrics such as accuracy and a classification report. Visualizations are generated to analyze model performance and predictions.

### Prediction on Single Image

An example of predicting on a new image is provided, where the model predicts the probability of melanoma cancer.

## File Structure

- `melanoma_cancer_customCNN.ipynb`: Jupyter Notebook containing the code implementation with custom CNN.
- `melanoma_cancer_resnet50.ipynb`: Jupyter Notebook containing the code implementation with resent.
- `melanoma_9914.jpg`: Example image for prediction.
- `test1`: Example image for prediction.
- `train`: Dataset for training.
- `test`: Dataset for test.# Melanoma_Cancer
