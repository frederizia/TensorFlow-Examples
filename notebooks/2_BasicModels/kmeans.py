
# coding: utf-8

# # K-Means Example
# 
# Implement K-Means algorithm with TensorFlow, and apply it to classify
# handwritten digit images. This example is using the MNIST database of
# handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
# 
# Note: This example requires TensorFlow v1.1.0 or over.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[2]:


# Import MNIST datan


# ### K-means clustering
# 
# - Unsupervised learning
# - Sort data into k clusters/groups according to data attributes
# - Clusters are formed due to similarities in those attributes 
# - A sort of mean value/prototype for each cluster/group is determined and data is sorted according to how close it is to this mean/prototype (centroid)
# - An initial centroid for each cluster is chosen and the data is sorted into clusters
# - The centroid is updated to reflect the cluster. This is an iterative process
# - The properties of each cluster can further be analysed to determine what defines each group

# In[8]:


# Parameters
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25 # The number of clusters
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)


# In[11]:


# Build KMeans graph
# Note FJ: Had to remove cluster_centers_vars, must be deprecated
# Note FJ:
# Use init_op to initialise clusters, until cluster_centers_initialized returns True
# Then use train_op
(all_scores, cluster_idx, scores, cluster_centers_initialized, 
init_op,train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()


# In[12]:


# Start TensorFlow session
sess = tf.Session()

# Run the initializer(Note FJ: Find initial centroids of clusters)
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))


# In[13]:


# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))


# Notes FJ:
# - Each centroid has multiple labels within it (from each associated data set?)
# - We choose the one used most often as label for centroid
# - Then look at test set and assign to 'correct' cluster
# - Assign label based on cluster
# - Check how oftern we were correct
# - Test accuracy presumably between 0 and 1 (1 perfect match)
# 
