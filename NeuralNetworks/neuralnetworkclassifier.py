import pandas
import sys
import numpy as np


LABEL = ''
FEATURES = []

class NeuralNetwork:
  def __init__(self, hidden_layer_size):
    # set initial weights and other stuff
    self.layer_depth = 3 # always 3 layers maybe unused
    self.hidden_layer_size = hidden_layer_size
    self.layer_sizes = [len(FEATURES) + 1, self.hidden_layer_size, self.hidden_layer_size, 1] #plus 1 for bias and 1 for last layer
    # creates matrix of weights from layer to layer in each 
    self.weights = [np.random.randn(y, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    print(self.weights)
  
  def stochastic_gradient_descent(self):
    pass

  def back_propigation(self, training_example):
    pass

  def predict_example(self, example):
    pass

# sigmoid
def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1.0 - sigmoid(x))

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

  nn = NeuralNetwork(3)