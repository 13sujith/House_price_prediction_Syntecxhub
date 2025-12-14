# House Price Prediction using Linear Regression

## 📌 Project Overview
This project predicts house prices using a Linear Regression model.
The California Housing dataset provided by scikit-learn is used for training and evaluation.

## 🎯 Objectives
- Load and explore a housing dataset
- Train a Linear Regression model
- Evaluate model performance using RMSE and R² score
- Interpret model coefficients
- Save the trained model and make predictions

## 📂 Dataset
- **Source**: California Housing Dataset (scikit-learn)
- **Features**:
  - MedInc (Median Income)
  - HouseAge
  - AveRooms
  - AveBedrooms
  - Population
  - AveOccup
  - Latitude
  - Longitude
- **Target**: Median House Price (in units of $100,000)

## 🛠 Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

## ⚙️ Model Used
- Linear Regression

## 📊 Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **R² Score**

## 📈 Results
- The model achieved moderate performance with a reasonable RMSE and R² score.
- Median income was found to be the most influential feature.

## 💾 Model Saving
The trained model is saved using `joblib` for future predictions.

## 🚀 How to Run
1. Clone the repository
2. Open the Jupyter Notebook
3. Run all cells in sequence

## 📌 Example Prediction
The model predicts house prices in units of hundred thousand dollars.

## 🧠 Conclusion
This project demonstrates the complete machine learning workflow from data loading to model deployment.

