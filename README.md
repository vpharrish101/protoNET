# Prototypical Networks for Few-Shot Learning
This repository contains a complete implementation of Prototypical Networks (ProtoNet) using PyTorch and Learn2Learn, designed for few-shot image classification on the CIFAR-FS dataset. The model is trained using an episodic framework and tested on unseen test episodes to evaluate generalization.

## Overview
Prototypical Networks classify query samples by computing their distances to class prototypes, which are the mean embeddings of support examples. The training process mimics the few-shot setting using `N-way K-shot` episodic tasks. It is loosely similar to K-NN
This implementation includes:-
- A 3-block convolutional neural network backbone
- Episode-based task generation and support-query splitting
- Training loop with real-time plotting of loss and accuracy
- Test-time evaluation using unseen classes
- Memory cleanup function for long-running training sessions


## Key Features
- Episodic training using support-query splits
- Lightweight CNN backbone for embedding 32x32 images
- Compatible with CIFAR-FS dataset
- Clear separation of training, validation, and test episodes
- Parameter tuning via a central configuration class
- Easy extension to other few-shot datasets

## Model Architecture
The backbone CNN consists of three convolutional blocks, each with:
- Conv2D -> BatchNorm -> ReLU -> Dropout -> MaxPool
  
It reduces a 32x32 RGB image to a 1024-dimensional flattened feature vector, which is then passed through a linear layer to obtain embeddings of the specified output size.

## Dataset
This implementation uses the CIFAR-FS dataset provided by Learn2Learn:
- Based on CIFAR-100, split into train, validation, and test classes
- Automatically downloaded on first run

## Results
<img width="1188" height="585" alt="image" src="https://github.com/user-attachments/assets/067de2a4-9a8f-4c4c-8dd9-e258c202ff63" />




