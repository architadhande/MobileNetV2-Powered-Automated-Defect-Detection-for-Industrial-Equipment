# MobileNetV2-Powered-Automated-Defect-Detection-for-Industrial-Equipment

**Author:** Archita Dhande  
**Date:** June 2025  

---

## Objective

The objective of this project was to develop an image classification system capable of categorizing images of industrial components as:
- **Defective**
- **Non-Defective**

Bonus goals included:
- Identifying specific types of defects (e.g. crack, cut, hole, print)
- Exploring optimizations for inference on hardware accelerators (GPU/TPU)

---

## 2. Dataset Description
Data: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
The dataset provided was the **Hazelnut dataset**, structured with images split into:
- **train** — containing only non-defective (`good`) images  
- **test** — containing five sub-categories:
  - `good`
  - `crack`
  - `cut`
  - `hole`
  - `print`

### 📊 Original Class Distribution:
| Set   | non-defective | defective |
|:------------|:----------------|:------------|
| Train | 391 | 0 |
| Test  | 40  | 70 |

---

## 3. Problem Faced

The original training set contained only **non-defective** images. As a result:
- The model overfitted to a single class.
- Achieved 100% training accuracy but only ~50% validation accuracy.
- Completely failed to generalize to unseen defective images.

---

## 4. Solution Applied

**Fix:**  
- Moved **10 defective images** from the test set into the training set.
- Created new `defective` and `non_defective` folders for binary classification.
- Recomputed class weights to address class imbalance during model training.

### Updated Class Distribution:
| Set   | non-defective | defective |
|:------------|:----------------|:------------|
| Train | 391 | 10 |
| Test  | 40  | 8 |

---

## 5. Data Preparation

- **Resizing** all images to **224×224**
- **Rescaling** pixel values to `[0, 1]`
- Applied **data augmentation** on the training set:
  - Random horizontal flips
  - Rotations up to 20°
  - Random zooms (up to 20%)

---

## 6. Model Development

Utilized **Transfer Learning** with **MobileNetV2** pre-trained on ImageNet.

### Model Architecture:
- **MobileNetV2 base model** (frozen)
- `GlobalAveragePooling2D`
- `Dropout(0.3)`
- `Dense(1, activation='sigmoid')`

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Evaluation Metric:** Accuracy  

---

## 7. Handling Class Imbalance

Computed **class weights** based on image counts to penalize errors in the minority class (defective) more heavily.

python
counts = [391, 10]
class_weights = {
    0: 401/2/391,  # non-defective
    1: 401/2/10    # defective
}


## 📈 Training & Validation

- **Epochs:** 20  
- **Early Stopping:** patience=5  

| Metric               | Value  |
|:---------------------|:--------|
| **Train Accuracy**       | 96.9%   |
| **Validation Accuracy**  | 83.3%   |

---

## Test Results & Evaluation

| Metric        | Value  |
|:---------------|:--------|
| **Test Accuracy**  | 83.33%  |
| **Precision**      | 69.44%  |
| **Recall**         | 83.33%  |
| **F1-Score**       | 75.76%  |

### Confusion Matrix  

| Actual \\ Predicted | Non-defective | Defective |
|:--------------------|:----------------|:-----------|
| **Non-defective**    | 0                | 8          |
| **Defective**        | 0                | 40         |

* Observation:**  
- Model predicted all images as defective  
- High recall for defective class  
- Zero precision for non-defective images due to class imbalance  

---

## Visualizations

- Plotted **Training vs Validation Accuracy/Loss**
- Displayed **Sample Predictions** with actual vs predicted labels  

---

## Insights & Challenges

- Data imbalance was the biggest issue affecting performance  
- Adding defective images improved recall, but wasn't sufficient for balanced classification  
- Transfer Learning with MobileNetV2 and data augmentation improved model generalization  
- Class weighting prioritized minority class performance  

---

## Lessons Learned

- Always verify class distributions before training  
- Severe class imbalance can critically affect model reliability  
- Transfer learning significantly accelerates convergence on small datasets  

---

## Future Work

- Implement **multi-class classification** for defect types  
- Use **synthetic data generation (GANs, SMOTE)** to balance defective samples  
- Explore **model quantization** and deploy on edge devices  
- Conduct **hyperparameter tuning** and test other architectures (EfficientNet, ResNet)

---

## Repository Structure

ML_Assignment/
├── data/
│ └── hazelnut/
├── hazelnut_train_binary/
├── hazelnut_test_binary/
├── src/
│ ├── data_preparation.py
│ ├── model_training.ipynb
│ ├── model_evaluation.py
│ └── visualization.py
├── report.md
├── requirements.txt
└── README.md
