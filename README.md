# Chest X-Ray Pneumonia Detection using DenseNet121

This project implements a **binary classification model** for pneumonia detection from chest X-rays using **transfer learning with DenseNet121**.

- **Frameworks**: TensorFlow / Keras
- **Dataset**: [Kermany’s Chest X-Ray Pneumonia dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Key Techniques**: Transfer Learning, Data Augmentation, Grad-CAM visualization
- **Performance**: Achieved ~95% training accuracy, ~87.5% validation accuracy.

## 📂 Repository Structure
```

ChestXray-Pneumonia-DenseNet/
│── data/                  # dataset (train/val/test)
│── notebooks/             # Jupyter notebooks
│── src/                   # source code
│   ├── densenet\_model.py # DenseNet121 model builder
│   ├── train.py           # training & evaluation script
│   └── gradcam.py         # Grad-CAM visualization
│── results/               # trained models, plots
│── README.md              # project documentation
│── requirements.txt       # dependencies
│── .gitignore             # ignored files

````

## 🚀 Quickstart

```bash
# Clone repo
 git clone https://github.com/<your-username>/ChestXray-Pneumonia-DenseNet.git
 cd ChestXray-Pneumonia-DenseNet

# Install requirements
 pip install -r requirements.txt

# Run training
 python src/train.py
````

## 📊 Results

* **Validation Accuracy**: \~87.5%
* **F1-Score (Test)**: \~0.50 (imbalanced dataset — further tuning needed)
* **Observation**: Model performs well but struggles with **small test set & noise in X-rays**.

## 🔍 Interpretability (Grad-CAM)

The repo includes **Grad-CAM visualizations** to highlight pneumonia-infected regions in X-rays.

## ⚠️ Note

While the training accuracy reached \~95%, **generalization on test data was lower** (\~50% F1). This highlights the **importance of robust validation, cross-validation, and dataset balancing** in medical ML.

---


