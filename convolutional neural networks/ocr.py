
# coding: utf-8

# In[1]:


import math
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


data = np.load('letters.npz')
images = data['images']
labels = data['labels']


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=0)


# In[4]:


learning_rate = 0.01
minibatch_size = 64
num_epochs = 20


# In[5]:


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 128], name="X")
Y = tf.placeholder(tf.float32, [None, 26], name="Y")

W1 = tf.get_variable("W1", [128, 64], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [64], initializer = tf.zeros_initializer())

W2 = tf.get_variable("W2", [64, 48], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [48], initializer = tf.zeros_initializer())

W3 = tf.get_variable("W3", [48, 26], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable("b3", [26], initializer = tf.zeros_initializer())

Z1 = tf.add(tf.matmul(X, W1), b1)
A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(A1, W2), b2)
A2 = tf.nn.relu(Z2)

Z3 = tf.add(tf.matmul(A2, W3), b3) 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[6]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    # number of training examples 
    m = X.shape[0] 
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle the records in training set (X, y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition the records after shuffling.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case 
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[7]:


init = tf.global_variables_initializer()
costs = []
seed = 0
m = X_train.shape[0]


# In[8]:


with tf.Session() as sess:
        
    # Run the initialization
    sess.run(init)
        
    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
 
            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 1 == 0:
            costs.append(epoch_cost)
                
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


    print("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))

