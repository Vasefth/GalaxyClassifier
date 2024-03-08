# GalaxyClassifier

# Classifying Galaxies Using Convolutional Neural Networks

## Overview

In this project, you'll build a neural network to classify images of galaxies into four categories: regular galaxies, galaxies with rings, galactic mergers, and irregular celestial bodies. Utilizing the Galaxy Zoo dataset, this project supports astronomical research by automating the annotation process of vast amounts of celestial image data.

## Dataset

The dataset is curated by Galaxy Zoo and consists of images categorized into four classes:
- Regular galaxies
- Galaxies with rings
- Galactic mergers
- Irregular celestial bodies

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn

## Installation

Ensure Python 3.x is installed. Install the required Python packages using pip:

```bash
pip install tensorflow scikit-learn
```

## Model Architecture

- The model includes convolutional layers with ReLU activation and max pooling layers, followed by a flattening layer, a dense layer with ReLU activation, and an output dense layer with softmax activation.
- The optimizer is Adam with a learning rate of 0.001.

## Training

- The images are preprocessed with `ImageDataGenerator` for rescaling and augmented to improve model generalization.
- The model is trained for 8 epochs with a batch size of 5.

## Evaluation

- The model's performance is evaluated using categorical accuracy and AUC metrics.
- Activations can be visualized to interpret the model's behavior.

## Usage

Execute the script to train the model and classify the galaxy images. The script outputs a summary of the model architecture, trains the model on the training set, and evaluates it on the test set.

## Contributing

Contributions are welcome! Fork the project to add features, improve the model, or contribute to the documentation.
