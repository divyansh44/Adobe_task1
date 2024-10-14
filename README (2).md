
# **Adobe Task 1 - Inter IIT Tech Meet 12.0**

## **Introduction**

This project is part of Task 1 for the Adobe problem statement in the Inter IIT Tech Meet 12.0. The goal is to build a combined model using CNNs and BERT transformers to handle multimodal inputs: images, text metadata, and OCR-extracted text from images. The model predicts a regression target (e.g., likes, scores) using extracted features from both image data and textual content. The solution efficiently preprocesses and trains models using PyTorch on a GPU.

---

## **Project Overview**

The model combines:
- **CNN**: To extract image embeddings.
- **BERT**: A transformer-based model to generate embeddings from text metadata and OCR-extracted text.

After concatenating these embeddings, the combined feature set is passed through fully connected layers to produce the final regression output.

---

## **Dataset**

The dataset includes:
- **Images**: URLs of images used as input for predicting the target.
- **Text Metadata**: Text from the 'content' column of the dataset.
- **OCR Text**: Extracted text from images using Optical Character Recognition (OCR).

The Adobe dataset was split into training and validation sets for model training and evaluation.

---

## **Model Architecture**

### 1. **CNN for Image Feature Extraction**
- **EfficientNet-b0** from the `timm` library is used for extracting image features.
- The model is pre-trained and fine-tuned on the target dataset.

### 2. **BERT for Text Embeddings**
- A **BERT base uncased** model (pre-trained) is used to generate embeddings (768 dimensions) for:
  1. **Text metadata**: General information related to the image.
  2. **OCR text**: Text extracted from the image.

### 3. **Combined Feature Set**
- The model concatenates:
  1. **CNN features from images** (128-dimensional embeddings).
  2. **BERT embeddings from metadata** (768-dimensional embeddings).
  3. **BERT embeddings from OCR text** (768-dimensional embeddings).

- The combined feature set is passed through:
  1. A **Linear layer**.
  2. A **Dropout layer** to reduce overfitting.
  3. Another **Linear layer**, mapping the output to a single neuron for the regression task.

---

## **Training Loop & Validation Strategy**

### **Optimizer**
- **AdamW** optimizer is used to update the model parameters.

### **Loss Function**
- **Mean Squared Error (MSE)** is used as the loss function for the regression task.

### **Epochs**
- The model is fine-tuned for **6 epochs**.

### **Scheduler**
- **StepLR** is used with a step size of 2, reducing the learning rate at regular intervals during training.

### **Batch Size**
- Both training and validation sets use a batch size of **16**.

---

## **Results**
- **Best model**: The best-performing model is saved after evaluation, based on validation performance (MSE and R² score).
  
---

## **Future Improvements**
- **Deeper analysis of data**: Further analysis and experimentation with data.
- **Hyperparameter Tuning**: Further tuning of learning rates, batch sizes, etc.
- **Ensemble Methods**: Experiment with combining this model with other models.
- **Data Augmentation**: Additional augmentations to improve robustness.

