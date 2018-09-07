
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
plt.scatter(data[:,0], data[:,1], c = labels.ravel()) 
plt.xlabel('X-axis') 
plt.ylabel('Y-axis') 
plt.title('Input data')



learning_rate = 0.05
epochs = 2000
n_features = data.shape[1]



tf.reset_default_graph()

#define placeholders
X = tf.placeholder(tf.float32, [4, n_features], name="X")
Y = tf.placeholder(tf.float32, [4, 1], name="Y")

# Initialize our weigts & bias
W = tf.Variable(tf.zeros([n_features, 1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)

Z = tf.add(tf.matmul(X, W), b)
prediction = tf.nn.sigmoid(Z)

