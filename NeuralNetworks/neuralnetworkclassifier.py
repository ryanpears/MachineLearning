import pandas
import sys
import numpy as np


LABEL = ''
FEATURES = []

class NeuralNetwork:
  def __init__(self, hidden_layer_size):
    # set initial weights and other stuff
    self.layer_depth = 4 # always 3 layers maybe unused plus a output
    self.hidden_layer_size = hidden_layer_size
    # weights are structured like weight[layer][to][from]
    if hidden_layer_size == None:
      # i think the paper neural net
      self.layer_sizes = [3, 3, 3, 1]
      self.weights = np.array([np.array([np.array([-1, -2, -3]), np.array([1, 2, 3])]), 
      np.array([np.array([-1, -2, -3]), np.array([1, 2, 3])]),
      np.array([np.array([-1, 2, -1.5])]) 
      ], dtype=object)
    else:
      self.layer_sizes = [len(FEATURES) + 1, self.hidden_layer_size, self.hidden_layer_size, 2] #plus 1 for bias and 1 for last layer
      self.weights = [np.random.randn(y-1, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    # creates matrix of weights from layer to layer in each 
    
    print("weoghts are ", self.weights)
  
  def stochastic_gradient_descent(self):
    pass

  def back_propigation(self, training_example):
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
    print("z_vectors ", z_vectors)
    print("activations ", activation_vectors)
    print("result ", result)
    # compute the loss
    loss_val = loss(result, y)
    print("loss is ", loss_val)
    # backward step
    # initialize d L / d w_{m n}^{h} = 0
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    print("weights init ", nabla_w)
    #maybe update the output layer weights first
    print("updating output layer")
    partial_cache = []
    for layer_index in reversed(range(len(self.weights))):
      layer = self.weights[layer_index]
      print("layer is ", layer)
      print("layer index is ", layer_index)
      if layer_index != len(self.weights)-1:
        for j in range(len(layer)):
          weights = layer[j]
          for i in range(len(weights)):
            all_next_layer_weights = self.weights[layer_index+1]
            # next_layer_weights = self.weights[layer_index+1][0][j+1]# plus 1 to skip bais weight
            next_layer_weights = []
            for next_weight_vec in all_next_layer_weights:
              next_layer_weights.append(next_weight_vec[j+1])
            # print("next layer weights", next_layer_weights)
            # print("activation_vectors[(layer_index)][j]", activation_vectors[(layer_index)][j+1])
            # print("activation_vectors[(layer_index -1)][i]", activation_vectors[(layer_index -1)][i])
            d_loss = 0
            for next_index, next_weight in enumerate(next_layer_weights):
              print("next_weight", next_weight)
              print("sigmoid_prime", sigmoid_prime_2(activation_vectors[(layer_index+1)][j+1]))
              print("activation ", activation_vectors[(layer_index)][i])
              print("i is ", i)
              if len(partial_cache) <= j:
                print("cool")
                partial_cache.append(loss_prime(result, y) * next_weight * sigmoid_prime_2(activation_vectors[(layer_index+1)][j+1]))
              print("partial cache is ", partial_cache)
              if layer_index == 0:
                d_loss += partial_cache[next_index] * next_weight * sigmoid_prime_2(activation_vectors[(layer_index+1)][j+1]) * activation_vectors[(layer_index)][i]
              else:
                d_loss += partial_cache[j] * activation_vectors[(layer_index)][i]
              # d_loss += nabla_w[layer_index+1][j][0] * activation_vectors[(layer_index)][i]
            # i think the derivative of futher up * weight * activtation
            # print("d_loss is ", d_loss)
            nabla_w[layer_index][j][i] = d_loss
            print(nabla_w)
      else: 
        # special case for last layer
        for j in range(len(layer)):
          weights = layer[j]
          print("weight is",  weights)
          print(activation_vectors[(layer_index -1)])
          for i in range(len(weights)):
            nabla_w[layer_index][j][i] = loss_prime(result, y) * activation_vectors[(layer_index)][i]
            # partial_cache[0].append(loss_prime(result, y) * weights[i])

    print(nabla_w)
    # for layer in range(1, len(nabla_w)):
    #   print("layer is ", layer)
    #   d_w_vec_i = nabla_w[-layer]
    #   print("", d_w_vec_i)
    #   for j, d_w_vec in enumerate(d_w_vec_i):
    #     print("d_w vec is ", d_w_vec)
    #     for i, d_w in enumerate(d_w_vec):
    #       dependent_vec = 1
    #       g_s = loss_prime(result, y) * dependent_vec * activation_vectors[-(layer+1)][i]
    #       print(g_s)
    #       nabla_w[-layer][j][i] += g_s
    # print(nabla_w)

    # for l in range(2, self.layer_depth):
    #   z_vec = z_vectors[-l]
    #   print(z_vec)
      
    # not sure of a better way to itterate weights
    # for weights_at_layer in self.weights[::-1]:
    #   print(weights_at_layer)
    #   for weight_vect in weights_at_layer:
    #     for weight in weight_vect:
    #       self.compute_partial_for(weight, z_vectors, activation_vectors)
   
    return

  def compute_partial_for(self, weight, z_vectors, activation_vectors):
     
    return


  def predict_example(self, example):
    # ugly but I'm confident
    for layer, weightsAtLayer in enumerate(self.weights):
      result_at_layer = [1] # for bias
      # print(example)
      for weights in weightsAtLayer:
        # print(weights)
        result = sigmoid(np.dot(weights, example))
        # print("result is", result)
        result_at_layer.append(result)
      example = result_at_layer
    
    #maybe a better way to do this since last layer we  don't do sigmoid
    result = sigmoid_inverse(result)
    print(result)
    return result

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

# file reading
def read_columns(file_path):
  with open(file_path, 'r') as file:
    columns = []
    for line in file:
      columnsInLine = line.strip().split(",")
      columns += columnsInLine
    return columns

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
  # train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  # test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  train_df.insert(loc=0, column="baisvalue", value=1)
  test_df.insert(loc=0, column="baisvalue", value=1)

  nn = NeuralNetwork(None)
  # nn.predict_example([1,1,1])
  nn.back_propigation([1,1,1,1])