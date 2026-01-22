# Gender Classification Using Support Vector Machine

## Project Overview
This project implements a **Gender Classification system** using **Support Vector Machine (SVM)** with multiple kernel functions. The goal is to evaluate and compare the performance of different SVM kernels and ensemble models for accurate gender prediction based on numerical features.

The project follows a complete machine learning pipeline including data loading, training, evaluation, and model comparison.

---

## Dataset
The dataset contains numerical features used to classify gender.

### Target Variable
- `gender`
  - Male
  - Female

### Features
- 7 numerical attributes (as provided in the dataset)

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Workflow
1. Load and inspect the dataset
2. Check for missing values
3. Split data into training and testing sets (80:20)
4. Train SVM models with different kernels
5. Evaluate using accuracy, confusion matrix, and classification report
6. Compare results with Random Forest models

---

## Models Implemented

### Support Vector Machine (SVM)
- Linear Kernel
- RBF Kernel
- Polynomial Kernel
- Sigmoid Kernel

### Ensemble Model
- Random Forest Classifier

---

## Model Performance Summary

| Model | Accuracy |
|------|---------|
| SVM (Linear) | **96.50%** |
| SVM (RBF) | **97.20%** |
| SVM (Polynomial) | **96.60%** |
| SVM (Sigmoid) | **46.65%** |
| Random Forest | **96.70%** |
| Random Forest (max_features=100) | **96.30%** |

---

## Best Performing Model
- **SVM with RBF Kernel**
- **Accuracy: ~97.2%**

---

## Sample Code – SVM (RBF Kernel)
```
from sklearn.svm import SVC

model = SVC(kernel='rbf')
model.fit(x_train, y_train)
```

## Model Evaluation

```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
Project Structure
├── gender_classification_v7.csv
├── gender_classification_using_svm.ipynb
├── README.md

## Future Enhancements
Hyperparameter tuning using GridSearchCV
Feature scaling and normalization
Cross-validation
Deployment using Flask or FastAPI
Comparison with deep learning models

## Author
Saravanavel E
AI & Data Science Student
GitHub: https://github.com/SaravanavelE

## License
This project is intended for academic and learning purposes.

y_pred = model.predict(x_test)
