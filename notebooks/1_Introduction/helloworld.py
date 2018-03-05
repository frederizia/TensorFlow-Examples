
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.

hello = tf.constant('Hello, TensorFlow!')


# In[3]:


# Start tf session
sess = tf.Session()


# In[4]:


# Run graph
print sess.run(hello)

