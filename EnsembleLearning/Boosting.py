import numpy
import DecisionTree
from DecisionTree.DecisionTree import weighted_entropy

LABEL = 'label'

def calculate_error(tree, df):
  return 1

def use_stump(stump, input):
  return 1

# true if val is possitive false otherwise
def use_adaboost(alphas, stumps, example):
  assert len(alphas) == len(stumps)
  val = 0
  for i in range(0, len(alphas)):
    val += alphas[i] * use_stump(stumps[i], example)
  return val > 0


def alpha_calc(error):
  return 0.5 * numpy.log((1-error)/error)

def AdaBoost(df, attributes, t):
  #TODO
  # for a round t
  alphas = []
  stumps = []
  
  for i in range(0, t):
    # make a decsion tree (h_t) of depth 1 
    stumps[i] = DecisionTree.ID3(df, attributes, weighted_entropy, 1)
    # error_t is the error on the training data
    e_t = calculate_error(stumps[i], df)
    # calculate its vote  wack alpha_t fuck
    alphas[i] = alpha_calc(e_t)
    #this will be updating the weight column in df
    # D_t+1(i) = D_t(i) /Z_t (exp(-alpha_t y_i * h_t(x_i)))
    D = []
    Z = 0
    for index, row in df.iterrows():
      value = -alphas[i] * row[LABEL] * use_stump(stumps[i], row)
      Z += value
      D.append(value)
    # hopefully
    for index, row in df.iterrows():
      D[index] = row['weights']/Z  * D[index]
    df['weights'] = D
    # loop
    
  # final is H_final(x) = sgn(sum_t alpha_t h_t(x))
  # really  unsure how to return a  function like this just returning pieces
  return alphas, stumps

if __name__ == "__main__":
  #want to just take in train.csv and test.csv
  print("ADAboost")
  # idea add a weight column to the dataframe do this
  # TODO parse the data and set weight to 1/total rows
  