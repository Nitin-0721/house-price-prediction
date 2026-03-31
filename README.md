# 🏠 House Price Prediction ML Project

## 📌 Overview
A complete end-to-end Machine Learning project that predicts house prices using Ridge Regression with 90% accuracy (R² = 0.9006).

## 🎯 Features
- Complete EDA with visualizations
- Data preprocessing & feature engineering
- Multiple ML models trained & compared
- Hyperparameter tuning with GridSearchCV
- Interactive Streamlit web app

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

## 📊 Model Performance
| Model | RMSE | R² Score |
|---|---|---|
| Ridge Regression | 0.1294 | 0.9006 |
| Linear Regression | 0.0029 | 0.8906 |
| Random Forest | 0.0032 | 0.8716 |
| Lasso Regression | 0.0038 | 0.8210 |
| Decision Tree | 0.0054 | 0.6286 |

## 📁 Project Structure
```
house-price-detection/
│
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Optimization.ipynb
├── models/             # Saved models
├── src/
│   └── app.py         # Streamlit app
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/house-price-prediction.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
cd src
streamlit run app.py
```

## 📈 Results
- **Best Model:** Ridge Regression (alpha=100)
- **R² Score:** 0.9006 (90% accurate!)
- **Training Data:** 1166 houses
- **Dataset:** Ames Housing Dataset

## Live Link https://house-price-prediction-gn07.streamlit.app/
 Ames Housing Dataset

## 👨‍💻 Author
- **Name:** Nitin
- **Project:** 100 Days of ML
