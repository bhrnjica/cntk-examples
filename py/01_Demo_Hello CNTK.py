# print helper must be the first line
from __future__ import print_function

# Import the relevant components
import sys
import os
import numpy as np
#
from collections import defaultdict

# Plot the data 
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
# %matplotlib inline # notebook relate directive

#import cntk library
import cntk as C

#cntk related components
import cntk.tests.test_utils
C.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable

# Ensure that we always get the same results
np.random.seed(0)

#network configuration ((2)I-(200)H-(1)O FFANN)
input_dim=2
output_dim = 1
neurons_num = 200

# training parameters
learning_rate = 0.003
minibatch_size = 20
num_samples_to_train = 20 #number of rows
iteration_number = 1000

# Helper function to generate 2D function f(x1,x2)=2x1+3x2
def generate_data_sample():
     # features creation  x1,x2
    X = ([
        [0.17,0.99],
        [0.82,1.00],
        [0.36,0.86],
        [0.82,0.31],
        [1.00,0.84],
        [0.40,0.87],
        [0.60,0.94],
        [0.81,0.29],
        [0.89,0.93],
        [0.49,0.83],
        [0.00,0.17],
        [0.16,0.78],
        [0.86,0.83],
        [1.00,0.08],
        [0.64,0.22],
        [0.32,0.60],
        [0.82,0.36],
        [0.95,0.16],
        [0.54,0.00],
        [0.71,0.84],
        ])
    # label creation f(x1,x2)
    Y = ([
       [295.],
       [424.],
       [300.],
       [247.],
       [416.],
       [309.],
       [367.],
       [239.],
       [418.],
       [315.],
       [53. ],
       [239.],
       [387.],
       [221.],
       [187.],
       [226.],
       [259.],
       [232.],
       [112.],
       [360.]
         ])

    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32)


# Define a utility that prints the training progress
def print_training_progress(trainer, mb, verbose=1):
    training_loss, eval_error = "NA", "NA"

    training_loss = trainer.previous_minibatch_loss_average
    eval_error = trainer.previous_minibatch_evaluation_average
    if verbose: 
     print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

#model creation
# FFNN with one input, one hidden and one output layer 
def create_model(features, hid_dim, out_dim):
    #perform some parameters initialization 
    with cntk.layers.default_options(init = cntk.glorot_uniform() ):
        #hidden layer with hid_def number of neurons and tanh activation function
        h1 = cntk.layers.Dense(hid_dim, activation= cntk.ops.tanh, name='hidLayer')(features)
        #output layer with out_dim neurons
        o = cntk.layers.Dense(out_dim)(h1)
        return o

def print_result(X,Y):
    #print data
    print("x1,x2; Y")
    print("________________________")
    print(" ")
    for i in range(0, len(Y)):
        print("({0},  {1});  {2}".format(str(round(X[i,0],2)),str(round(X[i,1],2)),str(round(Y[i,0],2))))

    print("________________________")
    print(" ")


# data definition
# features 3 columns with 10 rows
X,Y = generate_data_sample()
print_result(X,Y)



#input/output variable
feature = C.input_variable(input_dim, np.float32)
label = C.input_variable(output_dim, np.float32)

#NN model configuration
z = create_model(feature, neurons_num, output_dim)

# Define a dictionary to store the model parameters
mydict = {}

# learning rate parameter 
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch) 

#loss and evaluation function
loss = C.squared_error(z, label)
eval_error = C.squared_error(z, label)

# stochastic gradient decadent
learner = C.sgd(z.parameters, lr_schedule)

# Instantiate the trainer object to drive the model training
trainer = C.Trainer(z, (loss, eval_error), [learner])


# Three ways of learning NN
# 1. off-line learning: whole data set is passed to the training on each iteration 
# 2. on-line learning: only one instance is passed to the training on each iteration
# 3. mini-batch learning: 1 < batch size <= data set size is passed to the training on each iteration


# Definition of BatchSize - number of training instances seen by the model during a one iteration 
num_minibatches_to_train = int(num_samples_to_train / minibatch_size) * iteration_number

#training process
for i in range(0, num_minibatches_to_train):
        
    # Assign the mini-batch data to the input variables and train the model on the mini-batch
    trainer.train_minibatch({feature : X, label : Y})

    if i % 50 == 0:
        batchsize, loss, error = print_training_progress(trainer, i, verbose=1)
 
    
# Validation data set
Xval = np.array([[0.06,0.12],[0.03,0.13], [0.23,0.1], [0.32,0.5]], dtype=np.float32)
#
result = z.eval({feature : Xval})

#actual values of validation data set
Yval = np.array([[50],[47], [79], [199]])

print(" ")
print("___________________")
print("Validation model")
print_result(Xval,result)