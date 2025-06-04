readme_content = """
# 🚗 Car Price Prediction with 95% Accuracy

## 📌 Overview
This project focuses on predicting car prices using a dataset with structured automotive features. With a heavy emphasis on **feature engineering**, we transform raw data into powerful signals, resulting in a predictive model with over **95% accuracy**.

## 📊 Dataset
The dataset contains detailed information about various cars, including:
- Car names and brands
- Engine type and size
- Body style, drive wheels
- Mileage (MPG), horsepower, curb weight, etc.

## 🔍 Key Steps

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of car prices
- Correlation heatmaps to identify relationships
- Outlier detection and visualization

### 2. Feature Engineering
- Extracted car brand from `CarName`
- Created meaningful features (e.g., `power_to_weight_ratio`)
- Handled categorical variables with One-Hot Encoding
- Applied log transformation to reduce feature skewness

### 3. Model Building
- Implemented models using scikit-learn
- Pipelines were used to combine preprocessing and modeling
- Tested and validated with train-test split

### 4. Evaluation
- Evaluation Metrics: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
- Final model achieved **over 95% accuracy**

## 📈 Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## 💻 How to Run
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the notebook: