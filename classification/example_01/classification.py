""" 
Building a neural network for a multiclassification task. Addapted from various examples in the internet.
author: Ramon Cardias (2022) 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def create_model(my_learning_rate,input_shape):
        # Initiate a Sequential model
        model = tf.keras.models.Sequential()

        model.add(tf.keras.Input(shape=(input_shape,)))

        # Define the first hidden layer.   
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        # Define a dropout regularization layer. 
        model.add(tf.keras.layers.Dropout(rate=0.2))

        # Define the output layer. The units parameter is set to 4 because
        # the model must choose among 4 possible output values
        model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

        # Construct the layers into a model that TensorFlow can execute.  
        # Notice that the loss function for multi-class classification
        # is different than the loss function for binary classification.  
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

        return model

"""Train the model by feeding it data."""
def train_model(model, train_features, train_label, epochs,
                      batch_size=None, validation_split=0.1):

        history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                            epochs=epochs, shuffle=True,
                            validation_split=validation_split)

        # To track the progression of training, gather a snapshot
        # of the model's metrics at each epoch. 
        epochs = history.epoch
        hist = pd.DataFrame(history.history)

        return epochs, hist

if __name__ == '__main__':
	# Initialize parameter
	classes = 4
	m = 1000
	centers = [[-6, 3], [-3, -3], [3, 3], [6, -3]]
	std = 1.0
	X, y = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)

	# Scatter plot
	df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
	colors = {0:'red', 1:'blue', 2:'green', 3:'yellow'}
	grouped = df.groupby('label')
	fix, ax = plt.subplots(1,2)
	for key, group in grouped:
		group.plot(ax=ax[0], kind='scatter', x='x', y='y', label=key, color=colors[key])

	# Split data into train and validations set
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=1)
	
	# Creating model
        # Hyperparameters
	my_learning_rate = 0.001
	input_shape = 2
	validation_split = 0.2
	batch_size = X_train.shape[0] # In that case, the whole sample is used per epoch
	epochs = 200

	my_model = create_model(my_learning_rate,input_shape)

	epochs, hist = train_model(my_model, X_train, y_train,
                                   epochs, batch_size, validation_split) 

	# Evaluate against the test set.
	print("\n Evaluate the new model against the test set:")
	my_model.evaluate(x=X_val, y=y_val)

	# Vizualization
	y_pred = np.argmax(my_model.predict(X_val),axis=1)

	df = pd.DataFrame(dict(x=X_val[:,0], y=X_val[:,1], label=y_pred))
	colors = {0:'red', 1:'blue', 2:'green', 3:'yellow'}
	grouped = df.groupby('label')

	for key, group in grouped:
		group.plot(ax=ax[1], kind='scatter', x='x', y='y', label=key, color=colors[key])
	plt.show()
