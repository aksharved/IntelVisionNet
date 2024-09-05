# IntelVisionNet

This repository contains a CNN with computer vision implementation in the Intel Images dataset from Kaggle using PyTorch. The project includes data augmentation, batch normalization, and dropout for improved performance and generalization. It provides scripts for training and testing the model's performance.

IntelImageClassification/
│
├── models/                          # Directory to save and load models
│   └── intel_cnn_weights.pth         # Saved model weights
│
├── src/
│   ├── model.py                      # Definition of the CNN model
│   ├── train.py                      # Script for training the model
│   ├── test.py                       # Script for testing the model
│   └── data_loader.py                # Script for loading and processing data
│
└── README.md                         # Readme file for the project
