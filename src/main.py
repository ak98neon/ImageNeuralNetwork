import numpy

# number of input, hidden and output nodes
from src.network import NeuralNetwork

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.1
learning_rate = 0.1

# create instance of neural network
# n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n = NeuralNetwork.loadDump('neural_dump.txt')

# training_file = open("../mnist_dataset/mnist_train.csv", "r")
# training_data_list = training_file.readlines()
# training_file.close()
#
# epochs = 5
#
# for e in range(epochs):
#     for record in training_data_list:
#         all_values = record.split(",")
#         inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#         targets = numpy.zeros(output_nodes) + 0.01
#         targets[int(all_values[0])] = 0.99
#         n.train(inputs, targets)
#         pass
# pass

test_data_file = open("../mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

storecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)

    if label == correct_label:
        storecard.append(1)
    else:
        storecard.append(0)
        pass
pass

storecard_array = numpy.asarray(storecard)
print("Performance: ", storecard_array.sum() / storecard_array.size)
# n.dumpNetwork()
