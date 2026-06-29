# 🧠 MNIST Image Classifier using PyTorch

A beginner-friendly deep learning project built using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

> ⚠️ This is a learning project created to strengthen my understanding of the complete deep learning workflow using PyTorch.

---

## 📌 Project Overview

This project demonstrates the complete pipeline of building, training, and evaluating an image classification model.

The model is trained on the **MNIST handwritten digits dataset** and predicts which digit (0–9) is present in an input image.

Throughout this project, the focus was on understanding **how neural networks are built and trained**, rather than simply achieving the highest possible accuracy.

---

## 📂 Dataset

- **Dataset:** MNIST
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28 × 28 pixels
- **Number of Classes:** 10 (Digits 0–9)

---

## 🏗️ Model Architecture

The neural network consists of:

Input Image (28×28)

↓

Flatten Layer

↓

Linear Layer (784 → 128)

↓

ReLU Activation

↓

Linear Layer (128 → 10)

↓

Output (10 Classes)

---

## 📈 Training Pipeline

The project includes:

- Image preprocessing
- Data normalization
- DataLoader for mini-batch training
- Forward propagation
- Cross Entropy Loss
- Backpropagation
- Adam Optimizer
- Model evaluation on the test dataset

---

## 📊 Results

The repository also includes:

- 📉 Training Loss
- 📈 Test Accuracy
- 🔥 Confusion Matrix
- 🧠 Neural Network Architecture Diagram

---

## 📁 Project Structure

```text
Image Classification/
│
├── data/
├── data_test/
├── image/
│   ├── architecture.png
│   └── matrix_image.png
│
└── main.py
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib

---

## 🚀 How to Run

### Clone the repository

```bash
git clone <repository-link>
```

### Install dependencies

```bash
pip install torch torchvision matplotlib numpy pandas
```

### Run

```bash
python main.py
```

---

## 📚 What I Learned

Through this project I gained practical experience with:

- Creating custom neural networks using `nn.Module`
- Understanding tensors and image preprocessing
- Building training and evaluation loops
- Using optimizers and loss functions
- Working with GPU/CPU devices
- Evaluating classification models
- Visualizing model performance

---

## 🔮 Future Improvements

- Add Convolutional Neural Networks (CNNs)
- Implement model checkpoint saving
- Add learning rate scheduling
- Display sample predictions after testing
- Improve experiment tracking
- Deploy the model using Streamlit or Gradio

---

## 🎯 Purpose

This project is part of my learning journey toward becoming an **AI/ML Engineer specializing in Computer Vision, Robotics, and Physical AI**.

I believe the best way to learn AI is by building projects and understanding how each component works under the hood.

If you have any suggestions or feedback, feel free to connect or open an issue!
