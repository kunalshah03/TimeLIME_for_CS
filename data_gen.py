# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import os
# from sklearn.ensemble import RandomForestClassifier
# import lime.lime_tabular
# from imblearn.over_sampling import SMOTE
# from scipy import stats
# import time
# import itertools
# from mlxtend.frequent_patterns import apriori
# from mlxtend.preprocessing import TransactionEncoder## This file contains all tools needed to run planners.

# names = ['3dmol_3Dmol.js.csv', 'abinit_abinit.csv', 'alchemistry_alchemlyb.csv', 'Amber-MD_cpptraj.csv',
#          'Amber-MD_pytraj.csv', 'birkir_prime.csv', 'BOINC_boinc.csv', 'bustoutsolutions_siesta.csv',
#          'cclib_cclib.csv', 'chemfiles_chemfiles.csv']

# col_names = ['dates', 'monthly_open_issues', 'monthly_merged_PRs',
#        'monthly_closed_PRs', 'monthly_open_PRs', 'monthly_contributors',
#        'monthly_issue_comments', 'monthly_watchers', 'monthly_PR_comments',
#        'monthly_commits', 'monthly_PR_mergers', 'monthly_closed_issues',
#        'monthly_stargazer', 'monthly_forks', 'sina_score', 'paul_score',
#        'monthly_features', 'monthly_buggy_commits', 'developer_skill',
#        'license']
# array  = []
# for name in names:
#     # print(name)
#     n = name.split(".")[0]
#     # array.append(name)
#     split1 = str(n)+"_split_1.csv"
#     split2 = str(n)+"_split_2.csv"
#     split3 = str(n)+"_split_3.csv"
#     array.append(split1)
#     array.append(split2)
#     array.append(split3)

# print(array)


# Function to split and sort the CSV file
# def split_and_sort_csv(file_name, output_directory, date_column, split_count=2):
#     # Read the CSV file into a DataFrame
#     input_file = str("./data_sample/"+file_name)
#     df = pd.read_csv(input_file)

#     # Sort the DataFrame by the date column
#     df.sort_values(by=date_column, inplace=True)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     # Calculate the number of rows per split
#     split_size = (len(df) - 12) // 2

#     # Split and write the DataFrame into multiple CSV files
#     # for i in range(split_count):
#     #     start_idx = i * split_size
#     #     end_idx = (i + 1) * split_size if i < split_count - 1 else None
#     #     split_df = df[start_idx:end_idx]

#     #     output_file = os.path.join(output_directory, f'{file_name}_split_{i+1}.csv')
#     #     split_df.to_csv(output_file, index=False)
#     start_idx = 0
#     end_idx = split_size
#     split_df = df[start_idx:end_idx]
#     output_file = os.path.join(output_directory, f'{file_name}_split_1.csv')
#     split_df.to_csv(output_file, index=False)

#     start_idx = split_size
#     end_idx = split_size*2
#     split_df = df[start_idx:end_idx]
#     output_file = os.path.join(output_directory, f'{file_name}_split_2.csv')
#     split_df.to_csv(output_file, index=False)

#     split_df = df[-13:-1]
#     output_file = os.path.join(output_directory, f'{file_name}_split_3.csv')
#     split_df.to_csv(output_file, index=False)

#     print(f'Successfully split and sorted the CSV into {split_count} parts.')

# # Input CSV file
# # input_file = './data_sample/3dmol_3Dmol.js.csv'

# # Output directory for split files
# output_directory = 'data_new'

# # Date column to sort by
# date_column = 'dates'

# # Number of splits (default is 3)
# split_count = 2
# for name in names:
#     split_and_sort_csv(name, output_directory, date_column, split_count)


# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./data_sample/3dmol_3Dmol.js.csv')  # Replace 'your_data.csv' with the actual file path

# Convert the 'dates' column to datetime format
data['dates'] = pd.to_datetime(data['dates'])

# Sort the data by date
data = data.sort_values('dates')

# Create features
data['month'] = data['dates'].dt.month
data['year'] = data['dates'].dt.year

# Split the data into training and testing sets
train = data[:-12]
test = data[-12:]

# Select features and target variable
features = ['month', 'year', 'monthly_merged_PRs','monthly_closed_PRs','monthly_open_PRs','monthly_contributors','monthly_issue_comments','monthly_PR_comments','monthly_commits','monthly_PR_mergers','monthly_closed_issues','monthly_stargazer','monthly_forks','monthly_watchers','sina_score','paul_score','monthly_features','monthly_buggy_commits','developer_skill','license']  # Replace 'other_features' with relevant features
target = 'monthly_open_issues'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# print("X_train")
# print(X_train)
# print("y_train")
# print(y_train)
# print("X_test")
# print(X_test)
print("y_test")
print(y_test)

import xgboost as xgb
# or
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression

# Replace RandomForestRegressor with LinearRegression in the previous code
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
predictions_lr = lr_model.predict(X_test_scaled)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, predictions_lr)
# error = mean_absolute_percentage_error(y_test, predictions_lr)
print(predictions_lr)
print(f'Mean Squared Error (Linear Regression): {mse_lr}')

# Replace RandomForestRegressor with XGBoost or LightGBM
xgb_model = xgb.XGBRegressor(n_estimators=57, random_state=42)

xgb_model.fit(X_train_scaled, y_train)
predictions_xgb = xgb_model.predict(X_test_scaled)
# xgb_model.fit(X_train, y_train)
# predictions_xgb = xgb_model.predict(X_test)

# print("predictions_xgb")
# print(predictions_xgb)
# # Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, predictions_xgb)
print(f'Mean Squared Error (XGBoost): {mse_xgb}')


# # Train the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# # Make predictions for the next 12 months
predictions = rf_model.predict(X_test_scaled)

# # Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# # Visualize the results
# plt.plot(test['dates'], y_test, label='Actual')
# plt.plot(test['dates'], predictions, label='Predicted')
# plt.title('Monthly Open Issues Prediction')
# plt.xlabel('Date')
# plt.ylabel('Monthly Open Issues')
# plt.legend()
# plt.show()



