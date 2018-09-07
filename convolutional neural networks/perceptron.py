
# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Define input data 
data = np.array([[0.3, 0.2], [0.1, 0.4], [0.4, 0.6], [0.9, 0.5]], np.float32) 
labels = np.array([0, 0, 0, 1], np.float32) 
#converting the labels into 4x1 array
labels = np.reshape(labels, [4, 1])



# Plot input data 
plt.figure() 