**House Price Prediction**
This project implements a linear regression model, as well as decision tree and random forest models, to predict house prices based on various features such as square footage, number of bedrooms, bathrooms, and other relevant attributes.

**Table of Contents**
Introduction
Data
Installation
Usage
Model Training and Evaluation
Results

**Introduction**
Predicting house prices is a common problem in the field of data science. This project aims to build regression models to accurately predict the sale price of houses based on various features. We use linear regression, decision trees, and random forest regressors to achieve this goal.

**Data**
The dataset used in this project includes various features related to house characteristics and the target variable, SalePrice. The features include:

GarageArea
OverallQual
TotalBsmtSF
1stFlrSF
2ndFlrSF
LowQualFinSF
GrLivArea
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath
TotRmsAbvGrd
Calculated features: TotalBath and TotalSF

**Installation**
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt

**Usage**
Ensure the dataset CSV files (train.csv and test.csv) are in the project directory.
Run the script to preprocess the data, train the model, and make predictions:
bash
Copy code
python house_price_prediction.py
The script will output evaluation metrics and save the predictions to sample_submission.csv.

**Model Training and Evaluation**
The project uses three main regression models:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
The data preprocessing pipeline includes:

Imputing missing values with median values
Scaling features using standard scaling
We split the data into training and testing sets and evaluate the models using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Cross-validation is performed to ensure model stability.

**Results**
The predictions are saved to sample_submission.csv, which includes the house IDs and the predicted sale prices. Evaluation metrics, such as training MSE and RMSE, are printed to the console.
