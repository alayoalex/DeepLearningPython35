import NN_Nielsen.mnist_loader as ld
import NN_Nielsen.network as network
import NN_Nielsen.network2 as network2


def training_network():
    training_data, validation_data, test_data = ld.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def training_network2():
    training_data, validation_data, test_data = ld.load_data_wrapper()
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, 
        evaluation_data=test_data, lmbda = 5.0,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)