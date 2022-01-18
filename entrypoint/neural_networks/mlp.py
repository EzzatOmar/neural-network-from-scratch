import random
# MULTI LAYER PERCEPTRON

def activation_relu(x):
  """ ReLu activation function """
  if(x > 0): return x
  else: return 0

def activation_dev_relu(x):
  """ Return 1 for positvie numbers and else 0, the first derivative from ReLu """
  return 1 if(x > 0) else 0

def cost_mse(y, y_real):
  """ takes to vectors with y is the prediction and y_real the real values, returns the error """
  squareDiffs = [(y[i] - y_real[i])*(y[i] - y_real[i]) for i in range(len(y))]
  error = 0
  for diff in squareDiffs:
    error += diff
  error = round(error / len(y), 5)
  return error

def cost_dev_mse(y, y_real):
  """ Returns 2(y - y'), the first derivative from mse without summation """
  return round(2*(y - y_real), 5)

class mlp():
  debug = False
  learning_rate = 0.08
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
  input = None
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
    self.activation = lambda layer,z: z if(layer == self.hiddenLayers + 1) else activation_relu(z)
    self.activation_dev = lambda layer,z: 1 if(layer == self.hiddenLayers + 1) else activation_dev_relu(z)
    # self.activation = lambda layer,z: activation_relu(z)
    # self.activation_dev = lambda layer,z: activation_dev_relu(z)

    # if mse
    self.cost = cost_mse
    self.cost_dev = cost_dev_mse
    super()

  def initWeights(self):
    self.weights = []
    # input weights
    self.weights.append(self.createWeightMatrix(1,self.inputLength))
    # hidden weights
    for layer in range(self.hiddenLayers):
      self.weights.append(self.createWeightMatrix(self.units,len(self.weights[layer-1])))
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
    self.input = input
    lastOutput = input
    self.activationVector = []
    self.z = []
    for layer in range(self.hiddenLayers + 2):
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
        if(self.debug): print('  computeLayer x {} w {}'.format(x, w))
      b = biasVector[row]
      z += b
      if(self.debug): print('  computeLayer b {} z {}'.format(b, z))

      zVector.append(z)
      a = self.activation(layer, z)
      activation.append(a)
    return activation, zVector

  def measure(self, y, y_real):
    """ takes the prediction and the real value and returns the error """
    error = self.cost(y, y_real)
    return error

  def partialDerivativeCostOverActivation(self, layer, indexN, y_n = None):
    """ dC / da_n, caches the results """
    cache = self.cache['devActivations'][layer][indexN]
    if(self.debug): print('  partialDerivativeCostOverActivation', layer, indexN, y_n, cache)
    if(cache != None): return cache
    else:
      # last layer?
      if(layer == self.hiddenLayers + 1):
        a_n = self.activations[layer][indexN]
        cache = self.cost_dev(a_n, y_n)
      else:
        mSize = len(self.weights[layer+1])
        cache = 0
        for m in range(mSize):
          cache += self.partialDerivativeCostOverActivation(layer+1, m) \
                 * self.activation_dev(layer, self.z[layer+1][m]) \
                 * self.weights[layer+1][m][indexN]
      self.cache['devActivations'][layer][indexN] = cache
      return cache

  def partialDerivativeCostOverWeight(self, layer, indexM, indexN):
    """ dC / dw_mn, caches the results """
    if(self.debug): print('  partialDerivativeCostOverWeight', layer, indexM, indexN)
    cache = self.cache['devWeights'][layer][indexM][indexN]
    if(cache != None): return cache
    else:
      a_l_minus_1_n = self.input[indexN]
      if(layer > 0): a_l_minus_1_n = self.activations[layer-1][indexN]
      cache = self.partialDerivativeCostOverActivation(layer, indexM) \
            * self.activation_dev(layer, self.z[layer][indexM]) \
            * a_l_minus_1_n
      self.cache['devWeights'][layer][indexM][indexN] = cache
      return cache
    
  def partialDerivativeCostOverBias(self, layer, indexM):
    """ dC / dw_mn, caches the results """
    if(self.debug): print('  partialDerivativeCostOverBias', layer, indexM)
    cache = self.cache['devBiases'][layer][indexM]
    if(cache != None): return cache
    else:
      cache = self.partialDerivativeCostOverActivation(layer, indexM) \
            * self.activation_dev(layer, self.z[layer][indexM])
      self.cache['devBiases'][layer][indexM] = cache
      return cache

  def backprop(self, layer, y_real_vector):
    """ Runs the backprop algorithm recursively layer by layer """
    weights = self.weights[layer]
    biases = self.biases[layer]
    M = len(weights)
    N = len(weights[0])
    if(self.debug): print('Run backprop on layer {}'.format(layer))
    for m in range(M):
      y_m = None
      # last layer?
      if(layer == self.hiddenLayers + 1): y_m = y_real_vector[m]
      if(layer > 0):
        da_m = self.partialDerivativeCostOverActivation(layer, m, y_m )
        if(self.debug): print('  da_{}'.format(m), da_m)
      db_m = self.partialDerivativeCostOverBias(layer, m)
      if(self.debug): print('  db_{}'.format(m), db_m)
      for n in range(N):
        dw_mn = self.partialDerivativeCostOverWeight(layer, m, n)      
        if(self.debug): print('  dw_{}{}'.format(m,n), dw_mn)

    pass

  def gradientDescent(self, y_real_vector):
    self.initCache()
    # print('Gradient descent: we must differential these weights', ["len: {}".format(len(w)) for w in reversed(self.weights)], ' and biases', ["len: {}".format(len(b)) for b in reversed(self.biases)])
    lastLayer = self.hiddenLayers + 1
    if(len(self.weights[lastLayer]) != len(y_real_vector)): raise Exception("y_real_vector must be length of {}".format(len(self.weights[lastLayer])))
    for layer in reversed(range(0, self.hiddenLayers + 2)):
      if(self.debug): print("Layer {}".format(layer))
      # if(layer == lastLayer): print("Is last layer")
      self.backprop(layer, y_real_vector)

  def adjustWeightsAndBiases(self, deltaWeights, deltaBiases):
    for layer, weights in enumerate(deltaWeights):
      for m, row in enumerate(weights):
        for n, cell in enumerate(row):
          self.weights[layer][m][n] -= cell * self.learning_rate
          self.weights[layer][m][n] = round(self.weights[layer][m][n], 5)

    for layer, biases in enumerate(deltaBiases):
      for m, cell in enumerate(biases):
        self.biases[layer][m] -= cell * self.learning_rate
        self.biases[layer][m] = round(self.biases[layer][m], 5)

          
    # for layer in range(self.hiddenLayers + 2):
    #   # UPDATE BIASES      
    #   for idx,devBias in enumerate(self.cache['devBiases'][layer]):
    #     if(self.debug): print("bias update, old {}, delta -1 * {}".format(self.biases[layer][idx], devBias * self.learning_rate))
    #     self.biases[layer][idx] -= devBias * self.learning_rate
    #     self.biases[layer][idx] = round(self.biases[layer][idx], 5)
    #   # UPDATE WEIGHTS      
    #   for m,devWeightRow in enumerate(self.cache['devWeights'][layer]):
    #     for n,devWeightCell in enumerate(devWeightRow):
    #       if(self.debug): print("weight update, old {}, delta -1 * {}".format(self.weights[layer][m][n], devWeightCell * self.learning_rate))
    #       self.weights[layer][m][n] -= devWeightCell * self.learning_rate
    #       self.weights[layer][m][n] = round(self.weights[layer][m][n], 5)

  def epoch(self, dataset, EPOCH=3):
    """ dataset is a list of lists of lists [[x1,y1], [x2,y2], ...], with x and y vectors """
    self.input = input
    for epoch in range(EPOCH):
      print('EPOCH {}'.format(epoch))
      # NOTE errors not in use, can be used to get the weighted sum or errors
      errors = []
      deltaWeights = None
      deltaBiases = None
      for xVec,yVec in dataset:
        self.activations = []
        y = self.feedforward(xVec)
        print('FEED FORWARD {} -> {}'.format(xVec, y))
        error = self.measure(y, yVec)
        errors.append(error)
        print('ERROR', error)
        # print('BACKPROP')
        self.gradientDescent(yVec)
        # SAVE ACTIVATIONS
        if(deltaWeights == None): 
          deltaWeights = list(self.cache['devWeights'])
          deltaBiases = list(self.cache['devBiases'])
        else: 
          for layer, weights in enumerate(deltaWeights):
            for m, row in enumerate(weights):
              for n, cell in enumerate(row):
                cell += self.cache['devWeights'][layer][m][n]
          for layer, biases in enumerate(deltaBiases):
            for m, cell in enumerate(biases):
              cell += self.cache['devBiases'][layer][m]

      # print('ADJUST WEIGHTS AND BIASES', deltaWeights, deltaBiases)
      self.adjustWeightsAndBiases(deltaWeights, deltaBiases)
    for xVec,yVec in dataset:
      errors = []
      y = self.feedforward(xVec)
      error = self.measure(y, yVec)
      errors.append(error)
    totalError = 0
    for err in errors:
      totalError += err
    print('TOTAL ERROR', err)
    

  def testState(self):
    self.initCache()
    self.weights = [[[2]],[[2]]]
    self.biases = [[3],[4]]
    self.debug = True
    self.learning_rate = 0.1
  
  def printWeights(self):
    for layer in range(len(self.weights)):
      print("Weights layer {}".format(layer))
      weightMatrix = self.weights[layer]
      for row in weightMatrix: print(row)
      print("-------------------")
  