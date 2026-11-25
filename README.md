# Experiment 3: Regression Techniques in Business Data(Simple Linear Regression)

##  Aim
To implement Simple Linear Regression in Python to study and predict the relationship between two business-related variables.

## Dataset
A sample business dataset representing the relationship between **Advertising Expenditure** (in Lakhs) and **Sales Revenue** (in Crores). The goal is to analyze how changes in advertising investment affect sales performance.

| Index | Advertising (Lakhs) | Sales (Crores) |
| :---: | :---: | :---: |
| 0 | 18 | 12 |
| 1 | 22 | 18 |
| 2 | 25 | 20 |
| 3 | 30 | 24 |
| 4 | 35 | 28 |
| 5 | 40 | 35 |
| 6 | 45 | 38 |
| 7 | 50 | 40 |
| 8 | 55 | 42 |
| 9 | 60 | 45 |

## Algorithm / Steps Followed
1. Import required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Create and load the dataset into a Pandas DataFrame.
3. Display the dataset.
4. Select independent ($X$) and dependent ($Y$) variables and reshape $X$ into a 2D array.
5. Build a `LinearRegression` model using `sklearn.linear_model`.
6. Fit the model and calculate the intercept ($a$) and slope ($b$).
7. Predict $Y$ values for given $X$ values and evaluate performance using the $R^2$ score.
8. Plot the actual data points alongside the fitted regression line and predict sales for a new advertising budget.

##  Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `np.array()` | Create numerical arrays for regression input data |
| | `pd.DataFrame()` | Create or manage tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Simple Linear Regression model |
| | `model.fit()` | Train the linear regression model using $X$ and $Y$ values |
| | `model.predict()` | Predict $Y$ values for given $X$ inputs using the fitted model |
| | `model.score()` | Calculate model performance ($R^2$ score) |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plot scatter points (actual data points) |
| | `plt.plot()` | Plot regression line (predicted relationship) |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display the regression plot |

##  Program & Output
# 1. Import required libraries  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression   

# 2. Create the dataset  
advertising = np.array([18, 22, 25, 30, 35, 40, 45, 50, 55, 60])  
sales = np.array([12, 18, 20, 24, 28, 35, 38, 40, 42, 45])  

df = pd.DataFrame({  
    "Advertising (Lakhs)": advertising,  
    "Sales (Crores)": sales  
})  

# 3. Display the dataset  
display(df)  

# 4. Select independent (X) and dependent (Y) variables  
X = df[["Advertising (Lakhs)"]].values  
Y = df["Sales (Crores)"].values  

print("Independent variable (X):")  
print(X)  
print("\\nDependent variable (Y):")  
print(Y)  

# 5. Build and train the Simple Linear Regression model  
model = LinearRegression()  
model.fit(X, Y)  

# 6. Calculate intercept (a) and slope (b)  
a = model.intercept_  
b = model.coef_[0]   

print(f"Intercept (a): {a:.4f}")  
print(f"Slope (b): {b:.4f}")  
print(f"Regression equation: Sales = {a:.4f} + ({b:.4f} × Advertising)")  

# 7. Predict sales and evaluate model performance  
predicted_sales = model.predict(X)  
r2_score = model.score(X, Y)  

results = df.copy()  
results["Predicted Sales (Crores)"] = predicted_sales  
display(results.round(2))  
print(f"R² Score: {r2_score:.4f}")  

# 8. Plot actual data points and fitted regression line  
plt.figure(figsize=(9, 6))  
plt.scatter(X, Y, color="blue", label="Actual Sales")  
plt.plot(X, predicted_sales, color="red", linewidth=2, label="Regression Line")  
plt.title("Simple Linear Regression: Advertising vs Sales")  
plt.xlabel("Advertising Expenditure (Lakhs)")  
plt.ylabel("Sales Revenue (Crores)")  
plt.legend()  
plt.grid(True)  
plt.show()  

# 9. Predict sales for a new advertising budget  
new_advertising = np.array([[48]])   
new_sales_prediction = model.predict(new_advertising)  

print(f"Advertising Budget: {new_advertising[0, 0]} Lakhs")  
print(f"Predicted Sales: {new_sales_prediction[0]:.2f} Crores")  







##  Result
