import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('dataset.csv')

# Split the data into training and testing datasets
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=69)

# Write the training data to a new CSV file
train_data.to_csv('training_dataset.csv', index=False)

# Write the testing data to a new CSV file
eval_data.to_csv('evaluation_dataset.csv', index=False)
