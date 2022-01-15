import random
# MULTI LAYER PERCEPTRON

def activation_relu(x):
  """ ReLu activation function """
  if(x > 0): return x
  else: return 0

def cost_mse(y, y_real):
  """ takes to vectors with y is the prediction and y_real the real values, returns the error """
  squareDiffs = [(y[i] - y_real[i])*(y[i] - y_real[i]) for i in range(len(y))]
  error = 0
  for diff in squareDiffs:
    error += diff
  error = error / len(y)
  return error

class mlp():
  # weight matrixes by layer, inputWeight, hidden1, ..., outputWeight
  weights = []
  # biases vectors by layer
  biases = []
  def __init__(self, inputLength, hiddenLayers, units, outputLength, activation="relu", cost="mse"):
    self.inputLength = inputLength
    self.outputLength = outputLength
    self.hiddenLayers = hiddenLayers
    self.units = units

    self.initWeights()
    self.initBiases()
    # if relu
    self.activation = activation_relu
    # if mse
    self.cost = cost_mse
    super()

  def initWeights(self):
    self.weights = []
    # input weights
    self.weights.append(self.createWeightMatrix(self.units,self.inputLength))
    # hidden weights
    for l in range(self.hiddenLayers):
      self.weights.append(self.createWeightMatrix(self.units,self.units))
    # output weights
    self.weights.append(self.createWeightMatrix(self.outputLength,self.units))

  def initBiases(self):
    self.biases = []
    # input bias
    self.biases.append(self.createBiasVector(self.inputLength))
    # hidden biases
    for l in range(self.hiddenLayers):
      self.biases.append(self.createBiasVector(self.units))
    # output bias
    self.biases.append(self.createBiasVector(self.units))


  def createWeightMatrix(self, rows, cols):
    """ Return matrix with random numbers """
    return [[random.randint(0,100) for col in range(cols)] for row in range(rows)]

  def createBiasVector(self, length):
    """ Return vector with random numbers """
    return [random.randint(0,99) for i in range(length)]
  
  def feedforward(self, input):
    # activation(W * input + b)
    lastOutput = input
    for layer in range(len(self.biases)):
      output = self.runLayer(layer, lastOutput)
      lastOutput = output

  def runLayer(self, layer, input):
    weightMatrix = self.weights[layer]
    biasVector = self.biases[layer]
    outputLength = len(weightMatrix)
    output = []
    for row in range(outputLength):
      z = 0
      for col in range(len(weightMatrix[row])):
        if(layer ==3): print(row,col)
        x = input[col]
        w = weightMatrix[row][col]
        b = biasVector[col]
        z += w*x
      z += b
      a = self.activation(z)
      output.append(a)
    return output

  def printWeights(self):
    for layer in range(len(self.weights)):
      print("Weights layer {}".format(layer))
      weightMatrix = self.weights[layer]
      for row in weightMatrix: print(row)
      print("-------------------")
  