
# coding: utf-8

# In[1]:


# Basic Operations example using TensorFlow library.
# Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/


# In[2]:


import tensorflow as tf


# In[8]:


# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)
print a, b
# Note FJ: Need to run session to access the values, otherwise only TF object returned
print tf.Session().run(a)


# In[6]:


# Launch the default graph.
with tf.Session() as sess:
    print "a: %i" % sess.run(a), "b: %i" % sess.run(b)
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i" % sess.run(a*b)


# In[15]:


# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
# Note FJ: Can't 'run' a placeholder object without giving more info


# In[16]:


# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)


# In[18]:


# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    # Note FJ: input in form of dictionary
    print "Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3})
    print "Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3})


# In[19]:


# ----------------
# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])


# In[20]:


# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])


# In[21]:


# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)


# In[22]:


# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul. (Note FJ: This implies that the matrices before aren't actually 'created' as such 
# until the session is run)
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    result = sess.run(product)
    print result

