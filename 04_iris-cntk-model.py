# Blog post article https://bhrnjica.net/2017/11/12/using-cntk-2-2-and-python-to-learn-from-iris-data/
# 
import os, sys
import numpy as np
import cntk

# The data in the file must satisfied the following format:
#
# |labels 0 0 1 |features 2.1 7.0 2.2  - the format consist of 4 features and one 3 component hot vector
#
#represents the iris flowers 
def create_reader(path, randomize, input_dim, num_label_classes):
    
    #create the streams separately for the label and for the features
    labelStream = cntk.io.StreamDef(field='label', shape=num_label_classes, is_sparse=False)
    featureStream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

   #create deserializer by providing the file path, and related streams
    deserailizer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels = labelStream, features = featureStream)) 
    
    #create mini batch source as function return
    mb = cntk.io.MinibatchSource(deserailizer, randomize = randomize, max_sweeps = cntk.io.INFINITELY_REPEAT if randomize else 1)
    return mb


# Function that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        print ("Minibatch: {0}, Loss: {1:.4f}, Accuracy: {2:.2f}%".format(mb, training_loss,(1 - eval_error)*100))   
    return mb, training_loss, eval_error


#model creation
# FFNN with one input, one hidden and one output layer 
def create_model(features, hid_dim, out_dim):
    #perform some parameters initialization 
    with cntk.layers.default_options(init = cntk.glorot_uniform() ):
        #hidden layer with hid_def number of neurons and tanh activation function
        h1=cntk.layers.Dense(hid_dim, activation= cntk.ops.tanh, name='hidLayer')(features)
        #output layer with out_dim neurons
        o = cntk.layers.Dense(out_dim, activation = cntk.ops.softmax)(h1)
        return o



input_dim=4
hidden_dim = 50
output_dim=3

input = cntk.input_variable(input_dim)
label = cntk.input_variable(output_dim)

train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"data/trainData_cntk.txt")

# Create the reader to training data set
reader_train= create_reader(train_file,True,input_dim, output_dim)
z= create_model(input, hidden_dim,output_dim)
loss = cntk.cross_entropy_with_softmax(z, label)
label_error = cntk.classification_error(z, label)


# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = cntk.learning_parameter_schedule(learning_rate)
learner = cntk.sgd(z.parameters, lr_schedule)
trainer = cntk.Trainer(z, (loss, label_error), [learner])


# Initialize the parameters for the trainer
minibatch_size = 120 #mini batch size will be full data set
num_iterations = 20 #number of iterations 



# Map the data streams to the input and labels.
input_map = {
label  : reader_train.streams.labels,
input  : reader_train.streams.features
} 


# Run the trainer on and perform model training
training_progress_output_freq = 1

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_iterations)):
        # Read a mini batch from the training data file
        data=reader_train.next_minibatch(minibatch_size, input_map=input_map) 
        trainer.train_minibatch(data)
        batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq)
        if not (loss == "NA" or error =="NA"):
            plotdata["batchsize"].append(batchsize)
            plotdata["loss"].append(loss)
            plotdata["error"].append(error)

# Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["loss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["error"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()



# Read the training data
test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"data/testData_cntk.txt")

reader_test = create_reader(test_file,False, input_dim, output_dim)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 20
num_samples = 20
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):
    
    data = reader_test.next_minibatch(test_minibatch_size,input_map = test_input_map)
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


