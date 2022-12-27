import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Fit the encoder on the data
encoder.fit(['Aachen', 'Aarhus', 'Abilene'])

# Transform the data into integer values
names_encoded = encoder.transform(['Aachen', 'Aarhus', 'Abilene'])

# Get the corresponding string labels for the encoded values
names = encoder.inverse_transform(names_encoded)

# Create a dictionary mapping the encoded values to the string labels
names_dict = dict(zip(names_encoded, names))

names_str = str(names_dict)


# Initialize the LabelEncoder object
label_encoder = LabelEncoder()

# Load the model from the file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Set the feature names for the input data
feature_names = ["name"]

# Read the data from the "meteorites.csv" file into a dataframe
df = pd.read_csv("meteorites.csv")

# Encode the string names using LabelEncoder
df["name"] = label_encoder.fit_transform(df["name"])

# Split the data into input features (X) and output labels (y)
X = df[["name"]]
y = df[["reclat", "reclong"]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the model with the input data and feature names
X = X_train.values
model.fit(X, y_train)

# Create the main window
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Meteorite Coordinate Predictor")

# Create the meteorite name input field
meteorite_name_input = QLineEdit()

# Create the prediction label
prediction_label = QLabel()
prediction_label.setText(
    "Enter a meteorite name and click Predict to get the predicted coordinates.")

# Create the Predict button
predict_button = QPushButton("Predict")

label_encoder.fit(df["name"])

# Initialize the LabelEncoder object
encoder = LabelEncoder()

# Fit the encoder on the data
encoder.fit(df["name"])


def on_button_clicked():
    # Get the meteorite name from the input field
    meteorite_name = meteorite_name_input.text()

    # Encode the meteorite name using the fitted encoder
    names = [meteorite_name]
    names_encoded = encoder.transform(names)

    # Create the input features array
    X = np.array(names_encoded).reshape(-1, 1)

    # Use the model to make predictions on new data
    predictions = model.predict(X)

    # Extract the predicted latitude and longitude values
    latitude = predictions[:, 0]
    longitude = predictions[:, 1]

    # Update the prediction label with the predicted values
    prediction_label.setText(
        f"Predicted coordinates: Latitude = {latitude[0]:.2f}, Longitude = {longitude[0]:.2f}")


# Connect the button's clicked signal to the function
predict_button.clicked.connect(on_button_clicked)

# Create the layout for the input fields and button
input_layout = QHBoxLayout()
input_layout.addWidget(meteorite_name_input)
input_layout.addWidget(predict_button)


# Create the main layout
layout = QVBoxLayout()
layout.addLayout(input_layout)
layout.addWidget(prediction_label)

# Set the layout for the main window
window.setLayout(layout)

# Create the label to display the dictionary
label = QLabel()
label.setText(names_str)

# Add the label to the layout
layout.addWidget(label)

# Show the main window
window.show()

# Run the application
sys.exit(app.exec_())
