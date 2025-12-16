# Chest X-Ray Pneumonia Detection using DenseNet121

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo

Deployed link✨ - [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chest-x-ray-pneumonia-detection-ml.streamlit.app/)

This project implements a **binary classification model** for pneumonia detection from chest X-rays using **transfer learning with DenseNet121**.

---
## Pneumonia Detection using Chest X-ray Images

Pneumonia is an inflammatory condition primarily affecting the lungs, characterized by symptoms such as cough, chest pain, fever, and difficulty breathing. The goal of this project is to develop an automated system for detecting and classifying pneumonia in chest X-ray images.

<p align="center">
  <img width="485" height="480" alt="Chest X-ray Example" src="https://github.com/user-attachments/assets/646fa855-3ca4-4dd8-948a-9849067e0bdf" />
</p>


## Motivation

The motivation behind this project is to leverage artificial intelligence to accurately detect and classify pneumonia using medical imaging. Early and reliable detection can significantly aid healthcare professionals in providing timely diagnosis and treatment, reducing the dependency on manual radiology interpretations.

## Approach

This project applies transfer learning using DenseNet121 to build a binary classification model capable of distinguishing between pneumonia and normal cases. The model was implemented in Python using Keras/TensorFlow and trained in PyCharm with GPU acceleration.


**The training process included:**

1. Data preprocessing with augmentation (rotation, zoom, horizontal flips).
2. Model fine-tuning on chest X-ray images.
3. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
4. Visualization of model interpretability using Grad-CAM heatmaps.


## Key Technologies Used:

- Python
- TensorFlow / Keras
- DenseNet121 (Transfer Learning)
- scikit-learn
- Matplotlib & Seaborn
- Grad-CAM (for explainability)


## 📂 Repository Structure
```

Chest-X-Ray-Pneumonia-Detection/
│── data/                  # dataset (train/val/test)
│── notebooks/             # Jupyter notebooks
│── densenet_model.py     # DenseNet121 model builder
│── train.py               # training & evaluation script
│── gradcam.py             # Grad-CAM visualization
│── README.md              # project documentation
│── requirements.txt       # dependencies
│── .gitignore             # ignored files

````


- **Frameworks**: TensorFlow / Keras
- **Dataset**: [Kermany’s Chest X-Ray Pneumonia dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   It is organized into three folders: train, test, and val, each containing X-ray images (JPEG format) categorized into two classes: Pneumonia and Normal.
Total images: 5,863
Train: ~5,216 images
Validation: ~16 images
Test: ~624 images
- **Key Techniques**: Transfer Learning, Data Augmentation, Grad-CAM visualization
- **Architecture:** DenseNet121 (Pre-trained on ImageNet).
- **Explainability:** Integrated **Grad-CAM** heatmaps to visualize the regions of the lung the model focuses on for prediction.
- **Performance:** High training accuracy (~95%) with real-time inference capabilities.

---

## 📊 Dataset
The project uses the **Kermany’s Chest X-Ray Pneumonia dataset** sourced from Kaggle.
* **Total Images:** 5,863
* **Categories:** Normal vs. Pneumonia
* **Class Balance:** The dataset is imbalanced, with significantly more Pneumonia cases.
<img width="560" height="433" alt="866e9dc7-a61b-448b-afe3-8e2aaceefb95" src="https://github.com/user-attachments/assets/c1fc1cb6-1987-4fd8-b7ba-0d26b9e13367" />


---

## 🚀 How to Clone and Run the Project Locally

### 1. Clone the Repository
To view or run this project on your local machine, clone the repository using the following commands:

```bash
git clone https://github.com/BleeGleeWee/Chest-X-Ray-Pneumonia-Detection.git
cd Chest-X-Ray-Pneumonia-Detection
````

### 2. Install Dependencies

Ensure you have **Python** installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

To start the model training process, run:

```bash
python train.py
```

---

## 📈 Final performance & Results:

* **Training Accuracy:** 95.76%
* **Training Loss:** 0.1237
* **Validation Accuracy (Log):** 81.25%
* **Validation Loss:** 0.2429

## Detailed Evaluation Report
*Note: The detailed metrics below reflect a specific validation batch evaluation.*

```text
              precision    recall  f1-score   support

      Normal       0.40      0.25      0.31         8
   Pneumonia       0.45      0.62      0.53         8

    accuracy                           0.44        16
   macro avg       0.43      0.44      0.42        16
weighted avg       0.43      0.44      0.42        16

```


## 🔍 Model Interpretability (Grad-CAM)
We use Gradient-weighted Class Activation Mapping (Grad-CAM) to make the "black box" model transparent. The images below show the original X-ray alongside the heatmaps indicating areas indicative of Pneumonia.

---


## ⚠️ Note

While the training accuracy reached \~95%, **generalization on test data was lower** (\~50% F1). This highlights the **importance of robust validation, cross-validation, and dataset balancing** in medical ML.

---

## 🤝 Contributing

This project is open for contributions! If you have ideas to improve the model, add visualizations, or enhance the code, feel free to submit a **pull request** or open an **issue**.  

Please keep in mind:  
- This repository is intended for **educational and research purposes only**.  
- Any contributions should maintain the professional and ethical standards of AI in healthcare.  
- Ensure that new code is properly documented and reproducible.  

Welcoming improvements such as:  
- Better data preprocessing or augmentation techniques  
- New architectures or model enhancements  
- Additional visualizations (training curves, Grad-CAMs, confusion matrices)  
- Improved notebook explanations or tutorials

OR

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Thank you for helping make this project better! 🌟
---
