
# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn import datasets



# Define input data
#Loading the iris dataset
iris = datasets.load_iris()
data, labels = iris['data'], iris['target']
feature_names = iris['feature_names']
dataset = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= feature_names + ['target'])



X_train, X_test, y_train, y_test = train_test_split(dataset[feature_names], dataset[['target']], test_size=0.25, random_state=5)



y_train_one_hot = pd.get_dummies(y_train, columns=['target'])
y_test_one_hot = pd.get_dummies(y_test, columns=['target'])




learning_rate = 0.01
epochs = 1500
# Output neurons : 3
n_class = len(set(iris['target']))
# Hidden layer neurons : 5
n_hidden = 5
n_features = X_train.shape[1]



tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_features], name="X")
Y = tf.placeholder(tf.float32, [None, 3], name="Y")



W1 = tf.get_variable("W1", [n_features, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [n_hidden], initializer = tf.zeros_initializer())

W2 = tf.get_variable("W2", [n_hidden, 3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b2 = tf.get_variable("b2", [3], initializer = tf.zeros_initializer())

Z1 = tf.add(tf.matmul(X, W1), b1)
A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(A1, W2), b2)
prediction = tf.nn.softmax(Z2)

# Calculate the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z2, labels = Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[7]:



init = tf.global_variables_initializer()

costs = []

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: y_train_one_hot})
        
        if (epoch+1) % 500 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        costs = np.append(costs, c)
        
    predicted_labels = tf.to_float(tf.greater(prediction, 0.5))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, predicted_labels)))

    print ("Training Set Accuracy:", accuracy.eval({X: X_train, Y: y_train_one_hot}))
    print ("Test Set Accuracy:", accuracy.eval({X: X_test, Y: y_test_one_hot}))
    



plt.xlabel('number of epochs')
plt.ylabel('cost')
plt.plot(costs)

