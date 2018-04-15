# read_exp.py
import sys, os
import numpy as np
import cntk as C

# 
the_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"../data/data01.txt")

predictors = np.loadtxt(fname=the_file, dtype=np.float32, delimiter=" ", usecols=(1,2,3,4))
passengers = np.loadtxt(fname=the_file, dtype=np.float32, delimiter=" ", ndmin=2, usecols=[6]) # note!

input_dim = 4
hidden_dim = 12
output_dim = 1

input_Var = C.ops.input(input_dim, np.float32)
label_Var = C.ops.input(output_dim, np.float32)

# create and train the nnet object

np.set_printoptions(precision=2)
print("\n---- Predictions: ")
for i in range(len(predictors)):
  ipt = predictors[i]
  print("Inputs: ", end='')
  print(ipt, end='')
  # pred_passengers = nnet.eval( {input_Var: ipt} )
  pred_passengers = 1.0 + 0.12* i  # dummy prediction
  print("   Predicted: %0.2f \
   Actual: %0.2f" % (pred_passengers, passengers[i]))
print("----")

print("\nEnd experiment \n")