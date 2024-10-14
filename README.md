# Adobe_task1
Task 1 of the Adobe problem statement in the inter IIT tech meet 12.0
# Introduction
  This project aims to build a combined model using CNNs and BERT transformers for multimodal input (images, text metadata, and OCR text from images). The goal is to predict a target regression value (e.g.,   likes, scores) using features extracted from both image data and textual content. The project handles multiple datasets, preprocesses inputs, and efficiently trains models using PyTorch on a GPU.

# Project Overview
  This project implements a model combining:
    - CNN for image embeddings.
    - BERT base uncased transformer for generating embeddings from text metadata and OCR-extracted text.
  The model concatenates these embeddings to create a combined feature set and uses fully connected layers on these concatenated embeddings for the final output.

# Dataset
  Images: Input URLs for images used to predict the target.
  Metadata Text: Text under the 'content' column of the dataset.
  OCR Text: Extracted text from the image using Optical Character Recognition (OCR).
  The Adobe Dataset (with image URLs and text metadata) has been used. 

  The dataset was split into training and validation sets for model training and evaluation.

# Model Architecture
  >CNN for Image Feature Extraction
    -EfficientNet-b0 from the timm library is used to extract image features.
    -The model is pre-trained and fine-tuned on the target dataset.
  >BERT for Text Embeddings
    -BERT base uncased model (pre-trained) generates embeddings (768 dimensions) for:
      1)Text metadata: Information related to the image.
      2)OCR text: Extracted text from the image using OCR.
  Combined Feature Set
  The concatenation of:
    1)CNN features from images (128-dimensional embeddings).
    2)BERT embeddings from metadata (768-dimensional embeddings).
    3)BERT embeddings from OCR text (768-dimensional embeddings).
  The combined feature set is passed through linear layer followed by a dropout layer followed by a linear layer mapping to a single neuron for the regression output.
# Training Loop & Validation Strategy
  Optimizer: AdamW optimizer is used to optimize the model parameters.
  Loss Function: Mean Squared Error (MSE) is used for regression.
  Epochs: The model was fine tuned for 6 epochs.
  Scheduler: StepLR with a step size of 2 was used for learning rate decay.
  Batch Size: Batch sizes of 16 and 16 were used for train and validation sets.
  The training and validation loops use tqdm for progress tracking.





