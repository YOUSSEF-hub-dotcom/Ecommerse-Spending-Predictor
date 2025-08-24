from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("------------------------------")
print("The gain of this Dataset Known Who earned money more Website or App ?")
print("------------------------------")
df = pd.read_csv("C:\\Users\\youss\\Downloads\\Ecommerce Customers.csv")
pd.set_option('display.width',None)
print(df.head(51))
print("--------------------------------------")
print("===========>>> Basic Function:")
print("numbers of rows and columns:")
print(df.shape)
print("The name of columns:")
print(df.columns)
print("information about data:")
print(df.info())
print("Statistical Operations:")
print(df.describe().round())
print("number of frequency rows:")
print(df.duplicated().sum())
print("number of missing values:")
print(df.isnull().sum())
print("--------------------------------------")
print("===========>>> Cleaning Data:")
missing_percentage = df.isnull().mean() * 100
print("Percentage of Missing Values in columns:\n",missing_percentage)
sns.heatmap(df.isnull())
plt.title("No Missing Values in Dataset")
plt.show()
print("--------------------------------------")
print("===========>>> Exploration Data Analysis:")
plt.figure(figsize=(10,10))
sns.pairplot(df)
plt.show()
#-----------------------
print(df.drop(['Email','Address','Avatar'],axis=1,inplace=True))
print("The Relationship Between Variables")
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot=True)
plt.title("The Relationship Between Variables")
plt.show()
print("--------------------------------------")
sns.regplot(data=df,x='Length of Membership',y='Yearly Amount Spent')
plt.grid()
plt.show()
print("===========>>> Building Model:")
x = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
print(x)
y = df['Yearly Amount Spent']
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print("------------------------------------")
print("=========>>> Model Training and Prediction :")
model = LinearRegression()
print(model.fit(x_train,y_train))
y_predict = model.predict(x_test)
print(y_predict)
print("-------------------")
print(y_test.values)
print("------------------------------------")
print("=========>>> Building Evaluation:")
print("Mean Squard error:",mean_squared_error(y_test,y_predict))
print("Mean absolute error:",mean_absolute_error(y_test,y_predict))
print("model.score (Train):",model.score(x,y)) # 98%
print("model.score (Test):",model.score(x_test,y_test)) # 97
print("R2 Score:",r2_score(y_test,y_predict)) #97%
print("-----------------------------------")
print("=============>>> Coefficients:")
# we have four coefficient
print("The 4 coefficient we have")
print(model.coef_)
# y(Yearly Amount Spent) = m1 * x1(Avg) + m2 * x2(App) + m3 * x3(Website) + m4 * x4(Membership) + b
# The App is more than Website
print("The App earned money more than Website")
print('===============================================================================================================')
"""
# 🛒 Ecommerce Spending Predictor

A Machine Learning project that predicts customer yearly spending and identifies whether the **App** or the **Website** drives more revenue.  

### 🔹 Project Workflow:
- Data Cleaning & Exploration (EDA)
- Visualization of customer behavior
- Feature Engineering & Correlation Analysis
- Built a **Linear Regression Model**

### 🔹 Key Results:
- **Train Accuracy:** 98%  
- **Test Accuracy:** 97%  
- **R² Score:** 97%  
- The analysis shows that **the App contributes more to revenue compared to the Website**.  

### 🔹 Tech Stack:
- Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
------------------------------------------------------------------------------------------------
🟦 LinkedIn Project Section / Description

Ecommerce Spending Predictor 🛒

Built a Linear Regression model to analyze and predict yearly customer spending.

Conducted EDA, feature selection, and model evaluation.

Achieved 98% train accuracy, 97% test accuracy, and 97% R² score.

Key Insight: Customers generate more revenue via the App than the Website.
----------------------------------------------------------------------------------------------
🚀 Excited to share my new Machine Learning project:
Ecommerce Spending Predictor 🛒

In this project, I worked on an ecommerce dataset to analyze customer behavior and predict yearly spending.
✅ Conducted data cleaning, statistical analysis, and visualization.
✅ Built and trained a Linear Regression model.
✅ Achieved 98% Train Accuracy, 97% Test Accuracy, and 97% R² Score.
📊 Insight: The App contributes more to customer spending than the Website.

🔧 Tools & Libraries: Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn

👉 This project enhanced my skills in EDA, regression modeling, and interpreting business insights from data.
"""