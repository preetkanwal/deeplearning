
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

# Calculate the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

costs = []

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: data, Y: labels})
        
        if (epoch+1) % 500 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                "W=", sess.run(W), "b=", sess.run(b))
        costs = np.append(costs, c)
        
    predicted_labels = tf.to_float(tf.greater(prediction, 0.5))

    # Compute the accuracy
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, predicted_labels)))

    print ("Accuracy:", accuracy.eval({X: data, Y: labels}))


plt.plot(costs)
plt.xlabel('number of epochs')
plt.ylabel('cost')

