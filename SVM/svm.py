import math
import sys
import pandas
import numpy
from scipy import optimize
import random

FEATURES = []
LABEL = ""

# tune a??
def learning_rate_1(gamma_0, t):
  a = 2**10
  gamma_a = float(gamma_0/a)
  result = gamma_0 / (1 + gamma_a*t)
  return result


def learning_rate_2(gamma_0, t):
  return gamma_0/ (1 + t)


def primal_svm(df, C, gamma_0, learning_rate):
  """
  stochastic sub-gradient descent in primal form
  learning rate is a function
  """
  global FEATURES
  #initialize w_o
  #Note features plus 1 is for the b.
  w = [0] * (len(FEATURES)+1)
  # w_0 = [0] * (len(FEATURES)+1)
  epochs = 100
  N = len(df) # think this is number of samples
  # for epochs
  for i in range(0, epochs):
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    t = i
    gamma = learning_rate(gamma_0, t)
    # print(gamma)
    sub_w = [0] * (len(FEATURES)+1)
    #for each training example
    #TODO maybe clean up
    for index, row in df.iterrows():
      train_x = (row[:-1]).to_numpy()
      train_y = row[LABEL]
     
      decision_val = train_y * numpy.dot(w, train_x)
      if decision_val <= 1:
        xy = numpy.multiply(train_y, train_x)
        cn = C*N
        sub_w += - numpy.multiply(cn, xy)
      else:
        pass
    
    total_sub_w = w + sub_w/ numpy.linalg.norm(sub_w, 1)
    # old_w_sub = old
    w += w - numpy.multiply(gamma, total_sub_w)
    w = w / numpy.linalg.norm(w, 1)
    #check convergence
    # diff = w - old_w
    # print(sub_w)
    eps = i*0.01
    if numpy.all(numpy.absolute(total_sub_w) < eps) and i > 10:
      print(f"converged in {i} epochs")
      break

  return w


def dual_svm(X, Y, C):
  #maximize the problem I think??????
  row_count, col_count = X.shape
  inital_alphas = [random.random() * C] * row_count
  yArg = (Y,)#must be a tuple to pass in
  zero_constraint = {'type': 'eq', 'fun': dual_svm_zero, 'args': yArg }
  # bound each value by C
  b = (0, C)
  bnds = []
  for bound in range(len(inital_alphas)):
    bnds.append(b)
  #I think this works
  # idea batch this find the suport vectors only optimize points that are a support vector. 
  # batch like 200 at a time???
  # X transpos X 
  # numpy outer product
  result = optimize.minimize(dual_objective_function, 
                                    inital_alphas, 
                                    method='SLSQP',
                                    args=(X, Y),  
                                    bounds=bnds,
                                    constraints=zero_constraint,
                                    tol=0.001)
  optimal_alphas = result.x
  
  #w = sum alpha y xi
  w= [0] * col_count
  for index, x in enumerate(X):
    xy = numpy.multiply(Y[index], x)
    w += numpy.multiply(optimal_alphas[index], xy)
  return w


def dual_objective_function(alphas, X, Y):
  # X, Y = XY
  # print(alphas)
  row_count, col_count = X.shape
   # \sum \alphaI - 1/2 sum_j sum_i a_j a_i y_j y_i x_i dot x_j
  
  alpha_sum = 0
  for alpha in alphas:
    alpha_sum += alpha

  support_sum = 0
  for i, row_i in enumerate(X):
    x_i = row_i
    y_i = Y[i]
    a_i = alphas[i]
    for j, row_j in enumerate(X):
      x_j = row_j
      y_j = Y[j]
      a_j = alphas[j]
      support_sum += a_i * a_j * y_i * y_j * numpy.dot(x_i, x_j)
 
  return 0.5 * support_sum - alpha_sum

 
def dual_svm_zero(alphas, Y):
  """
  constraint that sum alpha y = 0
  """
  total = 0
  for index, y in enumerate(Y):
    total += alphas[index] * y
  return total


def gaussain_kernal_svm(X, Y, C, gamma):
  row_count, col_count = X.shape

  inital_alphas = [0] * row_count
  yArg = (Y,)#must be a tuple to pass in
  # not need to write a different 
  zero_constraint = {'type': 'eq', 'fun': dual_svm_zero, 'args': yArg }

  b = (0, C)
  bnds = []
  for bound in range(len(inital_alphas)):
    bnds.append(b)

  # i think instead of x_i dot x_j in the objective just do the gauassan
  result = optimize.minimize(dual_objective_function_guassian, 
                                    inital_alphas, 
                                    method='SLSQP',
                                    args=(X, Y, gamma),  
                                    bounds=bnds,
                                    constraints=zero_constraint,
                                    tol=0.001)
  optimal_alphas = result.x

  #w = sum alpha y xi
  w= [0] * col_count
  for index, x in enumerate(X):
    xy = numpy.multiply(Y[index], x)
    w += numpy.multiply(optimal_alphas[index], xy)
  return w, optimal_alphas


def dual_objective_function_guassian(alphas, X, Y, gamma):
  # X, Y = XY
  # print(alphas)
  row_count, col_count = X.shape
   # \sum \alphaI - 1/2 sum_j sum_i a_j a_i y_j y_i x_i dot x_j
 
  alpha_sum = 0
  for alpha in alphas:
    alpha_sum += alpha

  support_sum = 0
  for i, row_i in enumerate(X):
    x_i = row_i
    y_i = Y[i]
    a_i = alphas[i]
    for j, row_j in enumerate(X):
      x_j = row_j
      y_j = Y[j]
      a_j = alphas[j]
      # instead of dot do guassian exp norm x_i - x_j ^2 / gamma
      # print(x_i-x_j)
      norm_sq = numpy.linalg.norm(x_i - x_j) ** 2
      # print(norm_sq)
      guassian = math.exp(- norm_sq/gamma)
      support_sum += a_i * a_j * y_i * y_j * guassian
  
  return 0.5 * support_sum - alpha_sum

# file reading
def read_columns(file_path):
  with open(file_path, 'r') as file:
    columns = []
    for line in file:
      columnsInLine = line.strip().split(",")
      columns += columnsInLine
    return columns


# testing
def test_learned_weights(df, weights):
  incorrect = 0
  for index, row in df.iterrows():
    true_label = row[LABEL]
    test_data = row[:-1]
    prediction = 1 if numpy.dot(weights, test_data.to_numpy()) > 0 else -1
    if prediction != true_label:
      incorrect += 1
  print("error is ", incorrect/ len(df))


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
  train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  train_df.insert(loc=0, column="baisvalue", value=1)
  test_df.insert(loc=0, column="baisvalue", value=1)

  #TODO change
  allC = [100/873, 500/873, 700/873]
  # allC =[500/873]

  # print("problem 2a")
  
  # for C in allC:
  #   w = primal_svm(train_df, C, 2**-2, learning_rate_1)
  #   print(w)
  #   print("training error")
  #   test_learned_weights(train_df, w)
  #   print("test error")
  #   test_learned_weights(test_df, w)

  # print("problem 2b")
  # for C in allC:
  #   w = primal_svm(train_df, C, 2**2, learning_rate_2)
  #   print(w)
  #   print("training error")
  #   test_learned_weights(train_df, w)
  #   print("test error")
  #   test_learned_weights(test_df, w)
  
  # # problem 2c is a comparision

  #format data to numpy to help speed up optimzation
  train_df_x  = train_df.drop(columns=LABEL)
  train_df_y = train_df[LABEL]

  # I believe this is correct. due to the same error on the medium dataset
  # we get the same error on medium dataset so just use that.
  print("problem 3a")
  for C in allC:
    w = dual_svm(train_df_x.to_numpy(), train_df_y.to_numpy(), C)
    print(w)
    print("training error")
    test_learned_weights(train_df, w)
    print("test error")
    test_learned_weights(test_df, w)
  
  # print("problem 3b")
  # gammas = [0.1,0.5,1,5,100]
  
  # support_vec_index = {}
  # for C in allC:
  #   print("C is ", C)
  #   for gamma in gammas:
  #     # initalize suport vectors
  #     if math.isclose(C, 500/873, rel_tol=0.1):
  #       support_vec_index[gamma] = []

  #     print('gamma is', gamma)
  #     w, alphas = gaussain_kernal_svm(train_df_x.to_numpy(), train_df_y.to_numpy(), C, gamma)
  #     print("training error")
  #     test_learned_weights(train_df, w)
  #     print("test error")
  #     test_learned_weights(test_df, w)
  #     if math.isclose(C, 500/873, rel_tol=0.1):
  #         for index, alpha in enumerate(alphas):
  #           if not math.isclose(0, alpha, rel_tol=1e-5):
  #             support_vec_index[gamma].append(index)

  # print("problem 3c")
  # for index in range(0, len(gammas)-1):
  #   curr = numpy.array(support_vec_index[gammas[index]])
  #   next = numpy.array(support_vec_index[gammas[index+1]])
  #   print(f"count of overlapping support vectors between {gammas[index]} and {gammas[index+1]}")
  #   intersection = list(set(curr).intersection(next))
  #   print(len(intersection))
  
    
  
