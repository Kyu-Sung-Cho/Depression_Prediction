# **Mental Health Depression Prediction**

## **Project Overview**
This project predicts depression risk levels using advanced machine learning techniques. By combining **XGBoost**, **LightGBM**, and **CatBoost** models into a soft voting ensemble, the project achieves highly accurate and robust predictions on the given dataset.

---

## **Business Problem**
Depression is a significant global issue, requiring timely identification and intervention.  
### Project Goals:
- Build a predictive model to identify depression risk accurately.  
- Use engineered features to enhance model performance.  
- Leverage ensemble techniques to generalize well across unseen data.

---

## **Dataset**
**Source**: [Kaggle Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11)  
**Target Variable**: `Depression` (Binary: 0 = No Depression, 1 = Depression Risk)  

### **Feature Categories**:
- **Socio-Demographic**: Age, Gender, City, Degree  
- **Work/Study Factors**: Academic Pressure, Work Pressure, Work/Study Hours, CGPA  
- **Behavioral Factors**: Study Satisfaction, Job Satisfaction, Financial Stress, Sleep Duration  
- **Mental Health History**: Suicidal Thoughts, Family History of Mental Illness  

---

## **Preprocessing and Feature Engineering**

### **Preprocessing**
- One-hot encoding for categorical variables  
- Standard scaling for numerical features  
- Ensured consistency of features between train and test sets  

### **Feature Engineering**
Custom interaction terms and ratios:  

| **Feature Name**             | **Description**                                                            |
|------------------------------|----------------------------------------------------------------------------|
| `pressure_interaction`       | Product of Academic Pressure and Work Pressure                             |
| `satisfaction_interaction`   | Product of Study Satisfaction and Job Satisfaction                         |
| `stress_pressure_ratio`      | Financial Stress divided by Work Pressure (+1 to avoid zero division)      |
| `efficiency_ratio`           | Study Satisfaction divided by Work/Study Hours (+1 to avoid zero division) |
| `total_pressure`             | Sum of Academic Pressure, Work Pressure, and Financial Stress              |
| `mental_health_risk`         | Weighted pressure score factoring satisfaction inverse                     |

---

## **Model Architecture**

### **Ensemble Model Components**
The final model is a **soft voting ensemble** combining the following:  

1. **XGBoost**  
   - n_estimators=2000, learning_rate=0.01, max_depth=7  
   - gamma=0.2, subsample=0.8, colsample_bytree=0.8  

2. **LightGBM**  
   - n_estimators=2000, learning_rate=0.01, max_depth=7  
   - num_leaves=100, colsample_bytree=0.8  

3. **CatBoost**  
   - iterations=2000, learning_rate=0.01, depth=7  
   - subsample=0.8, colsample_bylevel=0.8  

**Voting Weights**:  
- XGBoost: **1.2**  
- LightGBM: **1.0**  
- CatBoost: **1.1**  

---

## **Performance**

### **Kaggle Scores**
| **Leaderboard** | **Score**    |
|------------------|-------------|
| Private Score   | **0.94093** |
| Public Score    | **0.94237** |

---

## **How to Run the Project**

1. **Setup Environment**  
   Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost
