import nn_nielsen.mnist_loader as ld
import nn_nielsen.network as network
import nn_nielsen.network2 as network2

# Nielsen Network 1
def training_network():
    training_data, validation_data, test_data = ld.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 0.5, test_data=test_data)


# Nielsen Network 2
def training_network2():
    training_data, validation_data, test_data = ld.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.75, 
        evaluation_data=test_data, lmbda = 1.0,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True,
        early_stopping_n=10)


#print()
#training_network()

print()
training_network2()