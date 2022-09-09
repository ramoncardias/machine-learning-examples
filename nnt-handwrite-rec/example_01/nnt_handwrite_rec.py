""" 
Building a neural network to recognize binary handwritting. Addapted from Andrew NG example.
author: Ramon Cardias (2022) 
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def load_data():
    """
    Load data
    """
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def sigmoid(x):
    """
    Sigmoid function     
    """
    return 1. / (1. + np.exp(-x))


if __name__ == '__main__':

	# load dataset
	X, y = load_data()

	print(X.shape)
        #Create the neural network
	model = Sequential(
	    [               
	        tf.keras.Input(shape=(400,)),    #specify input size
	        Dense(25,activation='relu'),
	        Dense(15,activation='relu'),
	        Dense(1,activation='sigmoid')
	    ], name = "binarywritting" 
	)

	model.summary() # Model summary

	model.compile(
	    loss=tf.keras.losses.BinaryCrossentropy(),
	    optimizer=tf.keras.optimizers.Adam(0.001),
	) # Compile the model using the Adam optimizer

	model.fit(
	    X,y,
	    epochs=20
	) # Fit the model

	# Vizualizing the predictions
	m, n = X.shape

	fig, axes = plt.subplots(8,8, figsize=(8,8))
	fig.tight_layout(pad=0.1)

	for i,ax in enumerate(axes.flat):
	    # Select random indices
	    random_index = np.random.randint(m)
    
	    # Select rows corresponding to the random indices and
	    # reshape the image
	    X_random_reshaped = X[random_index].reshape((20,20)).T
    
	    # Display the image
	    ax.imshow(X_random_reshaped, cmap='gray')
    
	    # Display the label above the image
	    ax.set_title(y[random_index,0])
	    ax.set_axis_off()
	    # Predict using the Neural Network
	    prediction = model.predict(X[random_index].reshape(1,400))
	    if prediction >= 0.5:
	        yhat = 1
	    else:
	        yhat = 0
    
	    # Display the label above the image
	    ax.set_title(f"{y[random_index,0]},{yhat}")
	    ax.set_axis_off()
	fig.suptitle("Label, yhat", fontsize=16)
	plt.show()
