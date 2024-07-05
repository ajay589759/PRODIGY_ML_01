import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the training data
traindf = pd.read_csv('train.csv')

# Select relevant numerical columns
numeric_df = traindf.select_dtypes(include='number')

# Display correlation with SalePrice
correlation_matrix = numeric_df.corr()
print(correlation_matrix['SalePrice'].sort_values(ascending=False))

# Select relevant features
req_tr = ["GarageArea", "OverallQual", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "SalePrice"]
selected_tr = traindf[req_tr].copy()

# Calculate total bathrooms and total square footage
selected_tr.loc[:, 'TotalBath'] = (selected_tr['BsmtFullBath'].fillna(0) + selected_tr['BsmtHalfBath'].fillna(0) + selected_tr['FullBath'].fillna(0) + selected_tr['HalfBath'].fillna(0))
selected_tr.loc[:, 'TotalSF'] = (selected_tr['TotalBsmtSF'].fillna(0) + selected_tr['1stFlrSF'].fillna(0) + selected_tr['2ndFlrSF'].fillna(0) + selected_tr['LowQualFinSF'].fillna(0) + selected_tr['GrLivArea'].fillna(0))

# Prepare the final training dataframe
train_df = selected_tr[['TotRmsAbvGrd', 'TotalBath', 'GarageArea', 'TotalSF', 'OverallQual', 'SalePrice']]

# Split the data into training and testing sets
train_set, test_set = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

# Separate features and labels
housing = train_set.drop("SalePrice", axis=1)
housing_labels = train_set["SalePrice"].copy()

# Create a pipeline for preprocessing
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Transform the training data
X_train = my_pipeline.fit_transform(housing)
Y_train = housing_labels

# Visualize the training data
sns.pairplot(train_df)
plt.tight_layout()
plt.show()

# Display correlation heatmap
sns.heatmap(train_df.corr(), annot=True)
plt.show()

# Load the test data
testdf = pd.read_csv("test.csv")

# Select relevant features for the test data
req_tst = ["GarageArea", "OverallQual", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd"]
selected_tst = testdf[req_tst].copy()

# Calculate total bathrooms and total square footage for the test data
selected_tst.loc[:, 'TotalBath'] = (selected_tst['BsmtFullBath'].fillna(0) + selected_tst['BsmtHalfBath'].fillna(0) + selected_tst['FullBath'].fillna(0) + selected_tst['HalfBath'].fillna(0))
selected_tst.loc[:, 'TotalSF'] = (selected_tst['TotalBsmtSF'].fillna(0) + selected_tst['1stFlrSF'].fillna(0) + selected_tst['2ndFlrSF'].fillna(0) + selected_tst['LowQualFinSF'].fillna(0) + selected_tst['GrLivArea'].fillna(0))

# Prepare the final test dataframe
test_df_unproc = selected_tst[['TotRmsAbvGrd', 'TotalBath', 'GarageArea', 'TotalSF', 'OverallQual']]
test_df = test_df_unproc.fillna(test_df_unproc.mean())

# Transform the test data
x_test = my_pipeline.transform(test_df.values)

# Train the model (RandomForestRegressor in this case)
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(Y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
print(f"Training MSE: {train_mse:.2f}, Training RMSE: {train_rmse:.2f}")

# Cross-validation
scores = cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

print_scores(rmse_scores)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Save the results to a CSV file
pred = pd.DataFrame(y_pred, columns=['SalePrice'])
sub_df = pd.read_csv('sample_submission.csv')
datasets = pd.concat([sub_df['Id'], pred], axis=1)
datasets.to_csv('sample_submission.csv', index=False)
