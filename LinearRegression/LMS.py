import sys
import numpy
from numpy.core.fromnumeric import prod
from numpy.lib.function_base import average, corrcoef


# TODO use a relative  path
sys.path.insert(0, "/Users/ryanpearson/Documents/CollegeHomework/Fall2021/MachineLearning/MachineLearning/DecisionTree")
# only using for csv parsing
from DecisionTree import get_training_data, read_columns

LABEL = 'label'

def vector_norm_diff(v1, v2):
  """
  should return norm(v1 - v2) 
  """
  diff = numpy.subtract(v1, v2)
  norm = numpy.linalg.norm(diff)
  return norm

def cost_function(df, weights):
  sum = 0
  for index, row in df.iterrows():
    # i think should exclude the label
    x_i = row.values[:-1]
    sum += (row[LABEL] - numpy.dot(weights, x_i)) ** 2
  return  sum/2

def cost_function_gradient(df, weights):
  gradient = []
  #need to not do the last
  for col in df.columns:
    if col != LABEL:
      partial_d = cost_function_partial(df, weights, col)
      # print("partial is", partial_d)
      gradient.append(partial_d)
  assert len(gradient) == len(weights)
  # print("gradient is ", gradient)
  return gradient

def cost_function_partial(df, weights, partial):
  sum = 0
  for index, row in df.iterrows():
    x_i = row.values[:-1]
    x_ij = row[partial]
    sum +=  (row[LABEL] - numpy.dot(weights, x_i)) * x_ij
  sum *= -1
  return sum




def batch_gradient_descent(df, learning_rate):
  # init weights vector.
  weights = []
  for col in df.columns:
    if col != LABEL:
      weights.append(0)
  # r starts at 1 then gets halfed
  print("created weight vector", weights)
  r = learning_rate
  round = 0
  while True:
    print('on round ', round)
    print("cost function value is ", cost_function(df, weights))
    round += 1
    # compute gradient of J(w) at w^t DeltaJ(w^t)
    gradient = cost_function_gradient(df, weights)
    # update w^t+1 = w^t - learning_rate * DeltaJ(w^t)
    new_weights = []
    for i in range(0, len(weights)):
      new_weights.append(weights[i] - r* gradient[i])

    #check convergence
    assert len(weights) == len(new_weights), "weight vectors differ"
    vector_diff = vector_norm_diff(new_weights, weights)
    if vector_diff < 1e-6:
      break
    # if not converging update and repeat
    #TODO this is the one thing I am unsure of because it should be degraded but idk by how much
    r /= 2
    weights = new_weights
    # print("weights is ", weights)

  return weights

def stochastic_gradient_descent(df, learning_rate):
  
  # init weights vector.
  weights = []
  for col in df.columns:
    if col != LABEL:
      weights.append(0)
  round = 0
  r = learning_rate
  while True:
    print('on round ', round)
    print("cost function value is ", cost_function(df, weights))
    round += 1
    # pretend this example is the whole example 
    sample = df.sample(n=1)
    # calculate gradient based on this single example
    gradient = cost_function_gradient(sample, weights)
    # update weight vector till within 1e-6 convergence
    new_weights = []
    for i in range(0, len(weights)):
      new_weights.append(weights[i] - r* gradient[i])

    #check convergence
    assert len(weights) == len(new_weights), "weight vectors differ"
    vector_diff = vector_norm_diff(new_weights, weights)
    if vector_diff < 1e-6:
      break
    # if not converging update and repeat
    #TODO this is the one thing I am unsure of because it should be degraded but idk by how much
    r /= 2
    weights = new_weights
    # print("weights is ", weights)
  return weights
  
def analytical_weight_calc(df):
  # w = (XX^T)^-1 * XY
  Y = df[LABEL].to_numpy()
  X = df.drop(columns=LABEL).to_numpy().transpose()
  XXt = numpy.matmul(X, X.transpose())
  XY = numpy.matmul(X, Y)
  w = numpy.matmul(numpy.linalg.inv(XXt), XY)
  return w

if __name__ == "__main__":
  print("cool")
  train_csv = sys.argv[1]
  test_csv = sys.argv[2]
  columns_txt = sys.argv[3]
  columns = read_columns(columns_txt)
  LABEL = columns[-1]

  train_df =  get_training_data(sys.argv[1], columns)
  test_df = get_training_data(sys.argv[2], columns)

  # add a column so that the bais can be part of the weight vector
  train_df.insert(loc=0, column="baisvalue", value=1)
  test_df.insert(loc=0, column="baisvalue", value=1)
  batch_learning_rate = 1/20
  stochastic_learning_rate = 1/32

  print("batch gradient descent")
  batch_weights = batch_gradient_descent(train_df, batch_learning_rate)
  print("batch weights are ", batch_weights)

  print("starting learning rate is ", batch_learning_rate)
  test_cost_value = cost_function(test_df, batch_weights)
  print("test cost value using batch weights is ", test_cost_value)

  print("stochastic gradient descent")
  stochastic_weights = stochastic_gradient_descent(train_df, stochastic_learning_rate)
  print("stochastic weights are ", stochastic_weights)

  print("starting learning rate is ", stochastic_learning_rate)
  test_cost_value = cost_function(test_df, stochastic_weights)
  print("test cost value using stochatics weights is ", test_cost_value)

  print("analytical solving")
  analytical_weights = analytical_weight_calc(train_df)
  print("analytical weights are ", analytical_weights)

  analytical_test_cost = cost_function(test_df, analytical_weights)
  print("test cost value using analytic weights is", analytical_test_cost)




