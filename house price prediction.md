```python
# ================================
# Step 1: Import Required Libraries
# ================================
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib


# ================================
# Step 2: Load Housing Dataset
# ================================
housing = fetch_california_housing()

# Create DataFrame for features
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# Target variable (House Price)
y = housing.target


# ================================
# Step 3: Data Exploration
# ================================
print("First 5 rows of dataset:")
print(X.head())

print("\nDataset Info:")
print(X.info())

print("\nStatistical Summary:")
print(X.describe())


# ================================
# Step 4: Feature Selection
# (Using all features)
# ================================
X_selected = X


# ================================
# Step 5: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)


# ================================
# Step 6: Train Linear Regression Model
# ================================
model = LinearRegression()
model.fit(X_train, y_train)


# ================================
# Step 7: Predict on Test Data
# ================================
y_pred = model.predict(X_test)


# ================================
# Step 8: Model Evaluation
# ================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)


# ================================
# Step 9: Interpret Coefficients
# ================================
coefficients = pd.DataFrame({
    "Feature": X_selected.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)


# ================================
# Step 10: Save the Trained Model
# ================================
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")


# ================================
# Step 11: Load Model & Example Prediction
# ================================
loaded_model = joblib.load("house_price_model.pkl")

# Example house (same feature order)
sample_house = [[8.5, 25, 6, 1, 300, 3, 34.2, -118.4]]

predicted_price = loaded_model.predict(sample_house)

print("\nPredicted House Price (in 100,000$ units):", predicted_price[0])
print("Predicted House Price in dollars: $", predicted_price[0] * 100000)

```

    First 5 rows of dataset:
       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \
    0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   
    1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   
    2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   
    3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   
    4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   
    
       Longitude  
    0    -122.23  
    1    -122.22  
    2    -122.24  
    3    -122.25  
    4    -122.25  
    
    Dataset Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   MedInc      20640 non-null  float64
     1   HouseAge    20640 non-null  float64
     2   AveRooms    20640 non-null  float64
     3   AveBedrms   20640 non-null  float64
     4   Population  20640 non-null  float64
     5   AveOccup    20640 non-null  float64
     6   Latitude    20640 non-null  float64
     7   Longitude   20640 non-null  float64
    dtypes: float64(8)
    memory usage: 1.3 MB
    None
    
    Statistical Summary:
                 MedInc      HouseAge      AveRooms     AveBedrms    Population  \
    count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000   
    mean       3.870671     28.639486      5.429000      1.096675   1425.476744   
    std        1.899822     12.585558      2.474173      0.473911   1132.462122   
    min        0.499900      1.000000      0.846154      0.333333      3.000000   
    25%        2.563400     18.000000      4.440716      1.006079    787.000000   
    50%        3.534800     29.000000      5.229129      1.048780   1166.000000   
    75%        4.743250     37.000000      6.052381      1.099526   1725.000000   
    max       15.000100     52.000000    141.909091     34.066667  35682.000000   
    
               AveOccup      Latitude     Longitude  
    count  20640.000000  20640.000000  20640.000000  
    mean       3.070655     35.631861   -119.569704  
    std       10.386050      2.135952      2.003532  
    min        0.692308     32.540000   -124.350000  
    25%        2.429741     33.930000   -121.800000  
    50%        2.818116     34.260000   -118.490000  
    75%        3.282261     37.710000   -118.010000  
    max     1243.333333     41.950000   -114.310000  
    
    Model Evaluation:
    RMSE: 0.7455813830127761
    R2 Score: 0.5757877060324511
    
    Model Coefficients:
          Feature  Coefficient
    0      MedInc     0.448675
    1    HouseAge     0.009724
    2    AveRooms    -0.123323
    3   AveBedrms     0.783145
    4  Population    -0.000002
    5    AveOccup    -0.003526
    6    Latitude    -0.419792
    7   Longitude    -0.433708
    
    Model saved as house_price_model.pkl
    
    Predicted House Price (in 100,000$ units): 4.059714322646123
    Predicted House Price in dollars: $ 405971.4322646123
    

    C:\Users\Admin\anaconda3\Lib\site-packages\sklearn\utils\validation.py:2739: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    


```python

```
