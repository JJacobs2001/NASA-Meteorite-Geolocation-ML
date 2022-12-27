Meteorite Landings Machine Learning
This code uses machine learning to predict the latitude and longitude values of meteorite landings based on the name of the meteorite. The data is read from a meteorites.csv file and is split into training and test sets. A linear regression model is trained on the training set and used to make predictions on the test set. The model's performance is evaluated by calculating the mean absolute error and root mean squared error of the predictions. Finally, the trained model is saved to a file using the pickle library.

Requirements
This code requires the following libraries:

pandas
numpy
scikit-learn
pickle

Usage
To run this code, use the following command:
python meteorites.py
The code will output the mean absolute error and root mean squared error of the model's predictions. The trained model will be saved to a file called model.pkl.

Data
The data for this code is a list of meteorite landing locations, with each location having a name, id number, reclat (latitude) and reclong (longitude) value. The data is read from a meteorites.csv file, which should be placed in the same directory as the code file.