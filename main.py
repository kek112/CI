from numpy import exp, array, random, dot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#hidden layer
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)


            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    #test function that uses the trained weights
    def test(self, input):
        output_from_layer1 = self.__sigmoid(dot(input, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer2

    def print_weights(self):
        print ("    Layer 1 : ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 :")
        print (self.layer2.synaptic_weights)

#test zum laufen bekommen
#ab welcher iteration hat es fertig gelernt, grafisch darstellen
#experimentieren mit anzahl neouronen und layer
#coole diagramme ploten

if __name__ == "__main__":

    random.seed(1)

    layer1 = NeuronLayer(5, 64)

    layer2 = NeuronLayer(5, 5)

    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()


    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    img0 = mpimg.imread("PixelBilder/Dreieck.png")
    img1 = mpimg.imread("PixelBilder/Dreieck_rotiert.png")
    img2 = mpimg.imread("PixelBilder/Quadrat.png")
    img3 = mpimg.imread("PixelBilder/Strich_horizontal.png")
    img4 = mpimg.imread("PixelBilder/Strich_vertikal.png")
    
    img1Rauschen= mpimg.imread("PixelBilder/Dreieck_rotiert_damaged.png")
    gray0 = np.reshape(rgb2gray(img0), -1)
    gray1 = np.reshape(rgb2gray(img1), -1)
    gray2 = np.reshape(rgb2gray(img2), -1)
    gray3 = np.reshape(rgb2gray(img3), -1)
    gray4 = np.reshape(rgb2gray(img4), -1)

    gray1Rauschen = np.reshape(rgb2gray(img1Rauschen), -1)

    #plt.imshow(gray1, cmap = plt.get_cmap('gray'))
    #plt.show()
    #print(gray1)
    
    training_set_inputs = array([gray0, gray1, gray2, gray3, gray4])
    training_set_outputs = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    hidden_state, output = neural_network.think(array(gray1Rauschen))
    print(output)

