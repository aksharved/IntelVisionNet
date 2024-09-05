# **IntelVisionNet**

This repository contains a CNN with computer vision implementation for the Intel Images dataset from Kaggle using PyTorch. The project includes data augmentation, batch normalization, and dropout for improved performance and generalization. It provides scripts for training, testing, and evaluating the model's performance. Model has about 83 percent accuracy.

## **Project Structure**

```bash
IntelImageClassification/
│   
│
├── src/
│   ├── model.py                      # Definition of the CNN model
│   ├── train.py                      # Script for training the model
│   ├── test.py                       # Script for testing the model
│   └── data_loader.py                # Script for loading and processing data
│
└── README.md                         # Readme file for the project
