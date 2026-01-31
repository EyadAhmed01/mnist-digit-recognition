# MNIST: From Linear Models to Deep Networks

A comprehensive machine learning project that explores MNIST digit classification, progressing from simple linear models to deep neural networks. This project implements various classification approaches using both NumPy (from scratch) and PyTorch.

## Overview

This project demonstrates the evolution of machine learning models for handwritten digit recognition on the MNIST dataset, starting with basic linear models and advancing to more sophisticated deep learning architectures.

## Features

- **Binary Classification**: Logistic regression for classifying digits 0 vs 1
- **Multi-class Classification**: Softmax regression for all 10 digit classes (0-9)
- **Dual Implementations**: 
  - NumPy implementations from scratch (manual gradient computation)
  - PyTorch implementations using built-in modules
- **Model Evaluation**: Comprehensive metrics including accuracy, confusion matrices, and per-class performance
- **Visualization**: Training/validation curves, loss plots, and confusion matrices

## Dataset

The project uses the MNIST dataset (`mnist_train.csv`), which contains:
- 784 pixel features (28x28 images flattened)
- 10 digit classes (0-9)
- Pixel values normalized to [0, 1] range

## Project Structure

```
.
├── main.ipynb              # Main Jupyter notebook with all implementations
├── mnist_train.csv         # MNIST training dataset (required)
├── assigment_2.pdf         # Assignment instructions
└── README.md               # This file
```

## Requirements

The project requires the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch` (PyTorch)
- `torchvision`
- `jupyter` (for running the notebook)

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision jupyter
```

## Usage

1. Ensure you have the `mnist_train.csv` file in the project directory
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab
3. Run the cells sequentially to:
   - Load and preprocess the MNIST data
   - Split data into train/validation/test sets
   - Train binary classification models (logistic regression)
   - Train multi-class classification models (softmax regression)
   - Evaluate models and visualize results

## Data Splits

The project uses the following data splits:
- **Test set**: 20% of the full dataset
- **Validation set**: 20% of the full dataset (25% of remaining 80%)
- **Training set**: 60% of the full dataset

## Models Implemented

### Binary Classification (0 vs 1)
- Logistic regression with manual gradient descent
- Binary cross-entropy loss
- Early stopping based on validation loss

### Multi-class Classification (0-9)
- Softmax regression (NumPy from scratch)
- Softmax regression (PyTorch)
- Cross-entropy loss
- Full-batch gradient descent

## Results

The notebook includes:
- Training and validation loss/accuracy curves
- Confusion matrices for test set evaluation
- Per-class accuracy metrics
- Comparison between NumPy and PyTorch implementations

## Notes

- All implementations use normalized pixel values (0-1 range)
- Models are trained with learning rate 0.01
- Early stopping is implemented to prevent overfitting
- Batch size: 64 for binary classification

## License

This project is part of an academic assignment.
