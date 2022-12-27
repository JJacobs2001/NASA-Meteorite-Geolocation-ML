import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Read the data from the "meteorites.csv" file into a dataframe
df = pd.read_csv("meteorites.csv")

# Encode the string names using LabelEncoder
label_encoder = LabelEncoder()
df["name"] = label_encoder.fit_transform(df["name"])

while True:
    # Split the data into input features (X) and output labels (y)
    df = df.dropna(subset=["reclat", "reclong"])
    X = df[["name"]]
    y = df[["reclat", "reclong"]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a linear regression model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set and evaluate the model's performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print the mean absolute error and root mean squared error of the model's predictions
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Save the model to a file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Sleep for a certain amount of time before starting the next iteration
    time.sleep(0.5)  # Sleep for 60 seconds before starting the next iteration
