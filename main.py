from numpy import exp, array, random, dot
import numpy as np
import matplotlib.image as mpimg

test_img_1 = 0
test_img_2 = 0

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        # randomize initial weights
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        # combine Layers into Network
        self.layer1 = layer1
        self.layer2 = layer2


    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)


            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # if self.test(test_img_1)[1] >0.9:
            #     print (iteration,"Ausreichende Erkennungsrate")
            #     break

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def think(self, inputs):
        output_from_layer1 = self.sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    #test function that uses the trained weights
    def test(self, input):
        output_from_layer1 = self.sigmoid(dot(input, self.layer1.synaptic_weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer2

    def print_weights(self):
        print ("    Layer 1 : ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 :")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

    random.seed(1)

    # Data for neural net
    input_layer= 64
    hidden_layer= 70
    output_layer= 8

    layer1 = NeuronLayer(hidden_layer, input_layer)

    layer2 = NeuronLayer(output_layer, hidden_layer)

    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Startgewichte: ")
    neural_network.print_weights()


    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    # Trainingsset
    img0 = mpimg.imread("PixelBilder/Dreieck.png")
    img1 = mpimg.imread("PixelBilder/Dreieck_rotiert.png")
    img2 = mpimg.imread("PixelBilder/Quadrat.png")
    img3 = mpimg.imread("PixelBilder/Strich_horizontal.png")

    img4 = mpimg.imread("PixelBilder/Strich_vertikal.png")
    img5 = mpimg.imread("PixelBilder/Strich_schräg_links.png")
    img6 = mpimg.imread("PixelBilder/Strich_schräg_rechts.png")
    img7 = mpimg.imread("PixelBilder/Kreis.png")


    gray0 = np.reshape(rgb2gray(img0), -1)
    gray1 = np.reshape(rgb2gray(img1), -1)
    gray2 = np.reshape(rgb2gray(img2), -1)
    gray3 = np.reshape(rgb2gray(img3), -1)
    gray4 = np.reshape(rgb2gray(img4), -1)
    gray5 = np.reshape(rgb2gray(img5), -1)
    gray6 = np.reshape(rgb2gray(img6), -1)
    gray7 = np.reshape(rgb2gray(img7), -1)

    # Testset
    img1Rauschen= mpimg.imread("PixelBilder/Dreieck_rotiert_damaged.png")
    img2Rauschen= mpimg.imread("PixelBilder/Quadrat_damaged.png")

    gray1Rauschen = np.reshape(rgb2gray(img1Rauschen), -1)
    gray2Rauschen = np.reshape(rgb2gray(img2Rauschen), -1)
    test_img_1 = gray1Rauschen
    test_img_2 = gray2Rauschen

    training_set_inputs = array([gray0,
                                 gray1,
                                 gray2,
                                 gray3,
                                 gray4,
                                 gray5,
                                 gray6,
                                 gray7])

    training_set_outputs = array([[1,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,1]
                                  ]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) Gewichtung nach Training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Testen eines nicht im Trainingsets enthaltenen Bildes")
    hidden_state, output = neural_network.think(array(gray1Rauschen))
    print(output)

    hidden_state, output = neural_network.think(array(gray2Rauschen))
    print(output)


