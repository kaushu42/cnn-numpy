# from layers import Convolutional, Max_pool, Fully_connected
# from forward import conv_forward, pool_forward, fc_forward
# from back import conv_back, pool_back, fc_back

class CNN(object):
    def __init__(self):
        self.learning_rate = None
        self.epoch = None
        self.batch_size = None
        self.X = None
        self.Y = None
        self.weights = None
        self.biases = None
        self.gradients = None

    def __initialize_weights(self):
        print('Initializing Weights.....')

    def __initialize_biases(self):
        print('Initializing Weights.....')

    def __forward(self):
        print('Doing Forward propagation.....')

    def __back(self):
        print('Doing Backpropagation.....')

    def fit(self, X, Y):
        self.__initialize_weights()
        self.__initialize_biases()
        print('Training.....')
        self.__forward()
        self.__back()
        print('Done training.....')

    def predict(self, X):
        print('Predicting Outputs for the test data.....')
        print('Done......')
