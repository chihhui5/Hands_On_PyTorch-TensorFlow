# 🖼️ CNN Image Classification: CIFAR-10 Hands-On

This project focuses on building and training a **Convolutional Neural Network (CNN)** to perform multi-class image classification. Using the classic **CIFAR-10 dataset**, this module demonstrates the transition from simple linear models to spatial-aware deep learning architectures.

## 📌 Project Objectives

* **Understand CNN Architecture**: Implement Convolutional, Pooling, and Fully Connected (Dense) layers.
* **Image Data Preprocessing**: Master data normalization and augmentation to improve model generalization.
* **Combat Overfitting**: Apply techniques like **Dropout** and **Batch Normalization** to ensure the model learns features, not noise.
* **Evaluation**: Analyze model performance using accuracy metrics and confusion matrices.

---

## 📊 Dataset: CIFAR-10

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The classes represent common objects:

* **Transport**: Airplane, Automobile, Ship, Truck.
* **Animals**: Bird, Cat, Deer, Dog, Frog, Horse.

---

## 🧠 Model Architecture

The implemented CNN follows a hierarchical structure to extract features from low-level edges to high-level object parts:

1. **Convolutional Layers**: Extract spatial features using learnable filters.
2. **Activation (ReLU)**: Introduces non-linearity into the model.
3. **Pooling Layers (Max Pooling)**: Reduces spatial dimensions (downsampling) to decrease computational load and prevent overfitting.
4. **Fully Connected Layers**: Interprets the extracted features to perform the final classification.

---

## 🛠️ Key Implementation Steps

### 1. Data Augmentation & Normalization

To make the model robust against variations in image position and lighting:

* **Random Cropping & Flipping**: Artificially increases the size of the training set.
* **Standardization**: Normalizing pixel values to a mean of 0 and standard deviation of 1 for faster convergence.

### 2. Training Pipeline

* **Loss Function**: Cross-Entropy Loss (standard for multi-class classification).
* **Optimizer**: Stochastic Gradient Descent (SGD) with momentum or Adam optimizer.
* **Validation**: Real-time tracking of validation accuracy to monitor for overfitting.

### 3. Performance Optimization

* **Dropout**: Randomly deactivating neurons during training to force the network to learn redundant representations.
* **Batch Normalization**: Stabilizing the learning process by re-centering and re-scaling layer inputs.

---

## 🏁 Results & Key Learnings

* **Spatial Feature Importance**: Learned why CNNs outperform standard Multi-Layer Perceptrons (MLPs) for image data by preserving local pixel relationships.
* **Hyperparameter Impact**: Observed how changes in learning rate and batch size significantly affect the training stability.
* **Model Depth**: Realized that adding more layers increases capacity but requires careful regularization to avoid performance degradation.

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/chihhui5/Hands_On_PyTorch-TensorFlow.git

```


2. Navigate to the Week 2 folder:
```bash
cd W2_CNN_&_CIFAR_10_classification

```


3. Open the `.ipynb` file in **Google Colab** or **Jupyter Notebook**.
4. Ensure you have a GPU runtime enabled (highly recommended for CNN training).
