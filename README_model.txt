Linear Regression Model for Predicting Latitude and Longitude of Meteorite Landings
This code trains a linear regression model using meteorite landing data to predict the latitude and longitude of future meteorite landings. The model is trained using the meteorites.csv file, which contains the following data:

name: the name of the meteorite
reclat: the latitude of the meteorite landing
reclong: the longitude of the meteorite landing
The model is trained using the name column as the input feature and the reclat and reclong columns as the output labels. The data is split into training and test sets, and the model is trained on the training set using a linear regression model from scikit-learn. The model's performance is evaluated using the mean absolute error and root mean squared error of the predictions on the test set.

To use the model to make predictions on new data, the names of the meteorites for which you want to predict the latitude and longitude must be encoded using the LabelEncoder object. The encoded names can then be used as the input features for the model's predict method, which will return an array of predicted latitude and longitude values.

The trained model is saved to the model.pkl file using the pickle module, which can be loaded and used to make predictions on new data.

Required Libraries
pandas
numpy
sklearn
pickle
Running the Code
To run the code, simply execute it using a Python interpreter. The model will be trained and saved to the model.pkl file, and the mean absolute error and root mean squared error of the model's predictions on the test set will be printed to the console.

To use the model to make predictions on new data, you will need to load the model from the model.pkl file and encode the names of the meteorites using the LabelEncoder object. You can then use the predict method of the model to make predictions on the encoded names.