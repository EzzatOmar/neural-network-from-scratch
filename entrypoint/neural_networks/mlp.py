import random
# MULTI LAYER PERCEPTRON

def activation_relu(x):
  """ ReLu activation function """
  if(x > 0): return x
  else: return 0

def activation_dev_relu(x):
  """ Return 1 for positvie numbers and else 0, the first derivative from ReLu """
  return x if(x > 0) else 0

def cost_mse(y, y_real):
  """ takes to vectors with y is the prediction and y_real the real values, returns the error """
  squareDiffs = [(y[i] - y_real[i])*(y[i] - y_real[i]) for i in range(len(y))]
  error = 0
  for diff in squareDiffs:
    error += diff
  error = error / len(y)
  return error

def cost_dev_mse(y, y_real):
  """ Returns 2(y - y'), the first derivative from mse without summation """
  return 2*(y - y_real)

class mlp():
  # activation function
  activation = None
  activation_dev = None
  # cost function
  cost = None
  cost_dev = None
  # weight matrixes by layer, inputWeight, hidden1, ..., outputWeight
  weights = []
  # biases vectors by layer
  biases = []
  # activations
  activations = []
  # z
  z = []
  # input
  input = []
  # cache
  cache = { "devActivations": [], "devWeights": [], "devBiases": [] }

  def __init__(self, inputLength, hiddenLayers, units, outputLength, activation="relu", cost="mse"):
    self.inputLength = inputLength
    self.outputLength = outputLength
    self.hiddenLayers = hiddenLayers
    self.units = units

    self.initWeights()
    self.initBiases()
    self.initCache()
    # if relu
    self.activation = activation_relu
    self.activation_dev = activation_dev_relu
    # if mse
    self.cost = cost_mse
    self.cost_dev = cost_dev_mse
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
    self.biases = [self.createBiasVector(len(weightMat)) for weightMat in self.weights]

  def initCache(self):
    """ Inits cache if lists the same dimension as weights, biases, activations filled with None """
    self.cache['devBiases'] = [[None for b in bias] for bias in self.biases]
    self.cache['devActivations'] = [[None for b in bias] for bias in self.biases]
    self.cache['devWeights'] = [[[None for col in row] for row in weightMat] for weightMat in self.weights]

  def createWeightMatrix(self, rows, cols):
    """ Return matrix with random numbers """
    return [[random.random() for col in range(cols)] for row in range(rows)]

  def createBiasVector(self, length):
    """ Return vector with random numbers """
    return [random.random() for i in range(length)]
  
  def feedforward(self, input):
    # activation(W * input + b)
    lastOutput = input
    self.activationVector = []
    self.z = []
    for layer in range(len(self.biases)):
      activation, z = self.computeLayer(layer, lastOutput)
      lastOutput = activation
      self.activations.append(activation)
      self.z.append(z)
    return lastOutput

  def computeLayer(self, layer, input):
    weightMatrix = self.weights[layer]
    biasVector = self.biases[layer]
    outputLength = len(weightMatrix)
    activation = []
    zVector = []
    for row in range(outputLength):
      z = 0
      for col in range(len(weightMatrix[row])):
        x = input[col]
        w = weightMatrix[row][col]
        z += w*x
      b = biasVector[row]
      z += b
      zVector.append(z)
      a = self.activation(z)
      activation.append(a)
    return activation, zVector

  def measure(self, y, y_real):
    """ takes the prediction and the real value and returns the error """
    error = self.cost(y, y_real)
    return error

  def partialDerivativeCostOverActivation(self, layer, indexN, y_m = None):
    """ dC / da_n, caches the results """
    cache = self.cache['devActivations'][layer][indexN]
    # print('partialDerivativeCostOverActivation', layer, indexN, y_m, cache)
    if(cache != None): return cache
    else:
      # last layer?
      if(layer == self.hiddenLayers + 1):
        a_n = self.activations[layer][indexN]
        cache = self.cost_dev(a_n, y_m)
      else:
        mSize = len(self.weights[layer+1])
        cache = 0
        for m in range(mSize):
          cache += self.partialDerivativeCostOverActivation(layer+1, m) \
                 * self.activation_dev(self.z[layer+1][m]) \
                 * self.weights[layer+1][m][indexN]
      self.cache['devActivations'][layer][indexN] = cache
      return cache

  def partialDerivativeCostOverWeight(self, layer, indexM, indexN, y_m):
    """ dC / dw_mn, caches the results """
    cache = self.cache['devWeights'][layer][indexM][indexN]
    if(cache != None): return cache
    else:
      # last layer?
      if(layer == self.hiddenLayers + 1):
        cache = self.cost_dev(self.activations[layer][indexM], y_m) \
              * self.activation_dev(self.z[layer][indexM]) \
              * self.activations[layer-1][indexN]
      else:
        cache = self.partialDerivativeCostOverActivation(layer, indexM, y_m) \
              * self.activation_dev(self.z[layer][indexM]) \
              * self.activations[layer-1][indexN] if(layer > 0) else self.input[indexN]
      self.cache['devWeights'][layer][indexM][indexN] = cache
      return cache
    
  def partialDerivativeCostOverBias(self, layer, indexM, y_m):
    """ dC / dw_mn, caches the results """
    print("partialDerivativeCostOverBias", layer, indexM, y_m)
    cache = self.cache['devBiases'][layer][indexM]
    if(cache != None): return cache
    else:
      # last layer?
      if(layer == self.hiddenLayers + 1):
        cache = self.cost_dev(self.activations[layer][indexM], y_m) \
              * self.activation_dev(self.z[layer][indexM])
      else:
        cache = self.partialDerivativeCostOverActivation(layer, indexM, y_m) \
              * self.activation_dev(self.z[layer][indexM])
      self.cache['devBiases'][layer][indexM] = cache
      return cache

  def backprop(self, layer, y_real_vector):
    """ Runs the backprop algorithm recursively layer by layer """
    weights = self.weights[layer]
    biases = self.biases[layer]
    M = len(weights)
    N = len(weights[0])
    print('Run backprop on layer {}'.format(layer))
    for m in range(M):
      y_m = None
      # last layer?
      if(layer == self.hiddenLayers + 1): y_m = y_real_vector[m]
      if(layer > 0):
        da_m = self.partialDerivativeCostOverActivation(layer, m, y_m )
        print('da_{}'.format(m), da_m)
      db_m = self.partialDerivativeCostOverBias(layer, m, y_m)
      print('db_{}'.format(m), db_m)
      for n in range(N):
        dw_mn = self.partialDerivativeCostOverWeight(layer, m, n, y_m)      
        print('dw_{}{}'.format(m,n), dw_mn)

    pass

  def gradientDescent(self, y_real_vector):
    self.initCache()
    print('Gradient descent: we must differential these weights', ["len: {}".format(len(w)) for w in reversed(self.weights)], ' and biases', ["len: {}".format(len(b)) for b in reversed(self.biases)])
    lastLayer = self.hiddenLayers + 1
    if(len(self.weights[lastLayer]) != len(y_real_vector)): raise Exception("y_real_vector must be length of {}".format(len(self.weights[lastLayer])))
    for layer in reversed(range(0, self.hiddenLayers + 2)):
      print("Layer {}".format(layer))
      if(layer == lastLayer): print("Is last layer")
      self.backprop(layer, y_real_vector)

    # dC/dw(layer,row,col)
    # dC/db(layer,row)
    pass
  
  def epoch(self, input, y_real_vector):
    self.input = input
    y = self.feedforward(input)
    print('FEED FORWARD', y)
    error = self.measure(y, y_real_vector)
    print('ERROR', error)
    self.gradientDescent(y_real_vector)
    print('BACKPROP')

  
  
  def printWeights(self):
    for layer in range(len(self.weights)):
      print("Weights layer {}".format(layer))
      weightMatrix = self.weights[layer]
      for row in weightMatrix: print(row)
      print("-------------------")
  