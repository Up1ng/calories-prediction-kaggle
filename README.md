# Calories Burn Prediction (Kaggle ML Project)

## Project Description

This project aims to predict the number of burned calories based on physiological and activity-related parameters using machine learning models.

The solution includes feature engineering, model comparison, hyperparameter tuning, and error analysis.

---

## Dataset

The dataset is taken from Kaggle competition:

Predict burned calories based on user characteristics and exercise data.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn

---

## Models Used

The following regression models were trained and compared:

- Linear Regression with polynomial features
- Random Forest Regressor
- XGBoost Regressor
- XGBoost with GridSearch tuning

---

## Feature Engineering

### Correlation filtering
Highly correlated features (> 0.9) were removed.

### Log transformation
Numerical features were transformed if it improved correlation with the target.

### Polynomial features
Second degree polynomial features were added.

### Scaling
StandardScaler was applied.

---

## Model Evaluation Metrics

Models were evaluated using:

- MAE
- RMSE
- R² Score

---

## Error Analysis

The project includes:

- Error distribution histogram
- True vs Predicted values scatter plot
- Feature importance analysis

---

## How to Run

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/calories-prediction-kaggle.git
cd calories-prediction-kaggle
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training script:

```bash
python src/train_model.py
```

---

## Results

Best results were achieved using tuned XGBoost model.

---

## Author

Student ML project created for practice and Kaggle competition.
