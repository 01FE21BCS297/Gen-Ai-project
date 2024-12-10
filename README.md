# Gen-Ai-project
# GAN-Based Human Face Generation

This project leverages **Generative Adversarial Networks (GANs)** to generate realistic human faces. Using the **CelebA dataset**, a rich collection of celebrity face images, the GAN model learns to produce lifelike and diverse human faces by capturing intricate patterns in facial features, expressions, and variations. 

---
## Installation

### Prerequisites
- **Python 3.6+**
- **PyTorch** (Deep learning framework)
- **torchvision** (Image processing utilities)
- **scipy** (For FID calculation)
- **tqdm** (For progress bars)
- **matplotlib** (For visualizing results)

Install the required packages:
```bash
pip install torch torchvision scipy tqdm matplotlib

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Generative Adversarial Networks (GANs) are a type of machine learning model used to generate new and realistic data by learning patterns from existing datasets. In this project, a GAN is trained on the **CelebA dataset** to generate human faces. The generator learns to create realistic images from random noise, while the discriminator learns to distinguish between real and generated images. Together, these networks improve iteratively to produce photorealistic human faces.

---

## Dataset

The **CelebA (CelebFaces Attributes Dataset)** is a large-scale dataset of over 200,000 celebrity face images with rich annotations for 40 attributes, including gender, age, expression, and accessories. The dataset is diverse in terms of:
- **Facial features**: Shape, expressions, and emotions.
- **Lighting and angles**: Indoor, outdoor, varied brightness and perspectives.
- **Ethnic and cultural representation**.

**Dataset Highlights**:
- Training Set: 162,770 images.
- Validation Set: 19,867 images.
- Test Set: 19,962 images.

Preprocessing steps include resizing images to \(128 \times 128\) and normalizing pixel values to \([-1, 1]\) for model compatibility.

---

## Model Architecture

### Generator
- Input: A random noise vector of fixed size (latent space).
- Output: A synthetic image resembling a human face.
- Layers: Fully connected and deconvolutional layers with Batch Normalization and ReLU activation.

### Discriminator
- Input: An image (real or generated).
- Output: Probability of the image being real.
- Layers: Convolutional layers with Leaky ReLU activation, leading to a binary classification output.

The **Generator** and **Discriminator** are trained alternately in a competitive framework to improve the quality of generated images.

---

## Evaluation Metrics

### 1. **Inception Score (IS)**
- Evaluates both the quality and diversity of generated images based on the output of an Inception model.
- Higher IS indicates more realistic and diverse images.

---

## Results

- **Generated Images**: The GAN model successfully generated diverse and realistic human faces, demonstrating the ability to replicate intricate facial details.
- **Evaluation Metrics**:
  - Inception Score: **1.025**
- Generated images exhibit a high degree of realism and diversity. However, minor artifacts and distortions in some images indicate areas for improvement.

---


