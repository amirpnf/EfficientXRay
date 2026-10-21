# EfficientXRay: Deep Learning for Pneumonia Detection

Efficient XRAY is an end-to-end Computer Vision project designed to classify chest X-ray images to assist in detecting pneumonia (normal vs. pneumonia cases). Built with **PyTorch**, this project focuses on model efficiency, data preprocessing, GPU acceleration, and reliable evaluation metrics tailored for medical imaging.

## Project Overview

Accurate and early detection of pneumonia from chest X-rays is critical for clinical decision-making. This project implements a modular Deep Learning pipeline to:

* Verify GPU hardware support for accelerated PyTorch training.
* Load and preprocess pediatric/adult chest radiograph datasets.
* Build, train, and evaluate convolutional neural network (CNN) classifiers.
* Generate evaluation graphs (Confusion Matrix, ROC Curve) to assess clinical reliability.

## Key Features & Stack

* **Framework:** PyTorch & torchvision
* **Core Skills:** Deep Learning Classification, Transfer Learning, GPU Computing, Data Preprocessing
* **Metrics & Evaluation:** Scikit-Learn, Matplotlib, Seaborn (Confusion Matrix, ROC Curve)
* **Hardware Acceleration:** CUDA / GPU verification script

## Repository Structure & File Descriptions

```
.
├── check_gpu.py         # Utility script to verify CUDA availability & GPU setup
├── data_loader.py       # Custom PyTorch Dataset & DataLoader pipeline
├── model.py             # Neural network architecture definitions
├── classifier.py        # Core classification logic and wrapper methods
├── train.py             # Standard training loop implementation
├── train_classfier.py   # Fine-tuning / training script for the classifier
├── main.py              # Main execution entry point for full pipeline execution
└── evaluate.py          # Model evaluation script (ROC Curve, Confusion Matrix)
```

### Detailed File Roles

* **`check_gpu.py`**: Ensures PyTorch is detecting your GPU (CUDA) correctly before running training jobs.
* **`data_loader.py`**: Handles dataset loading, image resizing, normalization, and PyTorch `DataLoader` creation.
* **`model.py`**: Contains the PyTorch neural network class and layer configurations.
* **`classifier.py`**: Encapsulates the high-level interface for making predictions and managing model outputs.
* **`train.py`**: Implements the forward/backward passes, loss computation, and optimizer steps.
* **`train_classfier.py`**: Specialized script for training and tuning the classifier component with full evaluation outputs.
* **`main.py`**: Orchestrates the workflow from data loading to model instantiation and training.
* **`evaluate.py`**: Computes performance metrics, plots ROC curves, and draws confusion matrices for model validation.

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed and optionally a CUDA-compatible GPU.

### 2. Hardware Verification

Run the GPU check script to verify PyTorch can utilize your Graphics Card:

```bash
python check_gpu.py
```

### 3. Usage

#### **Training the Model**

To launch the full training pipeline, run:

```bash
python main.py
```

Alternatively, to train the specific classifier module:

```bash
python train_classfier.py
```

#### **Evaluating the Model**

To compute metrics and display the ROC curve and confusion matrix on the test set:

```bash
python evaluate.py
```

> **Note on Medical Imaging Evaluation:** In medical diagnostics, minimizing **False Negatives** (high Recall/Sensitivity) is crucial to ensure infected cases are not missed.

## License

Distributed under the MIT License. See `LICENSE` for more information.
