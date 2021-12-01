import pandas
import sys
import numpy as np


LABEL = ''
FEATURES = []

class NeuralNetwork:
  def __init__(self, hidden_layer_size, random_weight_init):
    # set initial weights and other stuff
    self.layer_depth = 4 # always 3 layers maybe unused plus a output
    self.hidden_layer_size = hidden_layer_size
    # weights are structured like weight[layer][to][from]
    #TODO remove none option
    if hidden_layer_size == None:
      # i think the paper neural net
      self.layer_sizes = [3, 3, 3, 1]
      self.weights = np.array([np.array([np.array([-1, -2, -3]), np.array([1, 2, 3])]), 
      np.array([np.array([-1, -2, -3]), np.array([1, 2, 3])]),
      np.array([np.array([-1, 2, -1.5])]) 
      ], dtype=object)
    elif not random_weight_init:
      self.layer_sizes = [len(FEATURES) + 1, self.hidden_layer_size, self.hidden_layer_size, 2] #plus 1 for bias and 1 for last layer
      self.weights = [np.zeros((y-1, x)) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    else:
      self.layer_sizes = [len(FEATURES) + 1, self.hidden_layer_size, self.hidden_layer_size, 2] #plus 1 for bias and 1 for last layer
      self.weights = [np.random.randn(y-1, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    # creates matrix of weights from layer to layer in each 
    
    # print("weights are ", self.weights)
  
  def stochastic_gradient_descent(self, df, max_epoch, gamma_0):
    #given a set s = {x_i, y_i} = df
    #having initial weights set
    #for each epoch
    for epoch in range(max_epoch):
      #shuffle the data
      df = df.sample(frac=1).reset_index(drop=True)  
      gamma = learning_rate(gamma_0, epoch)
      # for each training example in df
      for index, row in df.iterrows():
        # treat example as the whole dataset
        #compute gradient
        gradient = self.back_propigation(row)
        # assert gradient.shape() == self.weights.shape()
        # update w = w - learning_rate  * gradient
        self.weights = self.weights - np.multiply(gamma, gradient)
      #TODO computer convergence
    print("learned weights ", self.weights)

  def back_propigation(self, training_example):
    # messy and really gross but it works
    x = training_example[:-1]
    y = training_example[-1]
    # forward pass similar to the prediction but storing useful data.
    z_vectors = [] # store the z vectors per layer before sigmoid  (maybe change)
    activation_vectors = [x] # store the activation vectors per layer
    # actualy pass
    #TODO this is a bit wonky for the last layer
    example = x
    for layer, weightsAtLayer in enumerate(self.weights):
      result_at_layer = [1] # for bias
      z_at_layer = [1] # for bias
      for weights in weightsAtLayer:
        z = np.dot(weights, example)
        z_at_layer.append(z)
        result = sigmoid(z)
        result_at_layer.append(result)

      z_vectors.append(z_at_layer)
      activation_vectors.append(result_at_layer)
      example = result_at_layer
    result = sigmoid_inverse(result)
    
    #init gradient
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    
    partial_cache = []
    for layer_index in reversed(range(len(self.weights))):
      layer = self.weights[layer_index]
      
      cache_index = len(self.weights) - layer_index -1
      partial_cache.append([])
     
      if layer_index != len(self.weights)-1:
        for j in range(len(layer)):
          weights = layer[j]
          for i in range(len(weights)):
            all_next_layer_weights = self.weights[layer_index+1]
            
            next_layer_weights = []
            for next_weight_vec in all_next_layer_weights:
              next_layer_weights.append(next_weight_vec[j+1])
           
            d_loss = 0
            for next_index, next_weight in enumerate(next_layer_weights):
              if len(partial_cache[cache_index]) <= j:
                partial = 0
                for index, cached_partial in enumerate(partial_cache[cache_index-1]):
                  partial += cached_partial * next_layer_weights[next_index+index] * sigmoid_prime_2(activation_vectors[(layer_index+1)][j+1])
                partial_cache[cache_index].append(partial)

              d_loss = partial_cache[cache_index][j] * activation_vectors[(layer_index)][i]
    
            nabla_w[layer_index][j][i] = d_loss
            
      else: 
        # special case for last layer
        for j in range(len(layer)):
          weights = layer[j]
          # add to cache
          partial_cache[cache_index].append(loss_prime(result, y))
          for i in range(len(weights)):
            nabla_w[layer_index][j][i] = loss_prime(result, y) * activation_vectors[(layer_index)][i]
            

    return nabla_w

  def predict(self, example):
    # ugly but I'm confident
    for layer, weightsAtLayer in enumerate(self.weights):
      result_at_layer = [1] # for bias

      for weights in weightsAtLayer:
        result = sigmoid(np.dot(weights, example))
        result_at_layer.append(result)

      example = result_at_layer
    #maybe a better way to do this since last layer we  don't do sigmoid
    result = sigmoid_inverse(result)
    return 1 if result > 0 else -1

# sigmoid
def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1.0 - sigmoid(x))

def sigmoid_prime_2(x):
  return (x) * (1.0 - (x))

def sigmoid_inverse(x):
  return np.log(x/(1.0 - x))

# Loss function
def loss(y, y_prime):
  return 0.5*((y - y_prime)**2)

def loss_prime(y, y_prime):
  return y - y_prime

#learning rate
def learning_rate(gamma_0, t):
  a = 2**10
  gamma_a = float(gamma_0/a)
  result = gamma_0 / (1 + gamma_a*t)
  return result

# file reading
def read_columns(file_path):
  with open(file_path, 'r') as file:
    columns = []
    for line in file:
      columnsInLine = line.strip().split(",")
      columns += columnsInLine
    return columns

# testing
def test_network(df, nn):
  incorrect = 0
  for index, row in df.iterrows():
    true_label = row[LABEL]
    test_data = row[:-1]
    prediction = nn.predict(test_data)
    # print(f"prediciton vs true {prediction} {true_label}")
    if prediction != true_label:
      incorrect += 1
  error = incorrect / len(df)
  print("error is ", error)

if __name__ == "__main__":
  train_csv = sys.argv[1]
  test_csv = sys.argv[2]
  columns_txt = sys.argv[3]

  columns = read_columns(columns_txt)
  LABEL = columns[-1]
  FEATURES = columns[:-1]

  train_df = pandas.read_csv(train_csv, names=columns, index_col=False)
  test_df = pandas.read_csv(test_csv, names=columns, index_col=False)

  # format data
  # not sure if I need this
  train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  train_df.insert(loc=0, column="baisvalue", value=1)
  test_df.insert(loc=0, column="baisvalue", value=1)

  nn = NeuralNetwork(None, True)
  nn.back_propigation([1,1,1,1])
  WIDTHS = [5, 10, 25, 50, 100]
  for width in WIDTHS:
    nn = NeuralNetwork(width, False)
    nn.stochastic_gradient_descent(train_df, 10, 0.05)
    print("training error")
    test_network(train_df, nn)
    print("test error")
    test_network(test_df, nn)
