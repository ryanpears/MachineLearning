import random
import sys
import os
import pandas
import numpy
from numpy.core.fromnumeric import prod
from numpy.lib.function_base import average, corrcoef


sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DecisionTree')))
from DecisionTree import DecisionTree, weighted_entropy, entropy, ID3, get_training_data, read_columns, set_label

LABEL = 'label'
# likely use
LAGGING= [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]


def calculate_error(stump, df):
  # i think this is right
  total = 0
  for index, row in df.iterrows():
    if use_stump(stump, row) != row[LABEL]:
      total += row['weight']

  return total

def use_stump(stump, row):
  # print(stump)
  
  feature = stump.feature

  if stump.children and row[feature] in stump.children.keys():
    return use_stump(stump.children[row[feature]], row)
  elif stump.children:
    # random guess if we've never seen the feature?
    return use_stump(random.choice(list(stump.children.values())), row)
  else:
    #leaf node
    assert (feature == -1 or feature == 1)
    return feature 

# true if val is possitive false otherwise
def use_voted_stumps(alphas, stumps, example):
  assert len(alphas) == len(stumps)
  val = 0
  for i in range(0, len(alphas)):
    val += alphas[i] * use_stump(stumps[i], example)
  return 1 if val > 0 else -1


def alpha_calc(error):
  return numpy.log((1-error)/error) * 0.5

def AdaBoost(df, attributes, t):
  alphas = []
  stumps = []

  for i in range(0, t):
    # make a decsion tree (h_t) of depth 1 
    stumps.append(ID3(df, attributes, weighted_entropy, 1))
    # error_t is the error on the training data
    e_t = calculate_error(stumps[i], df)
   
    alphas.append(alpha_calc(e_t))

    #this will be updating the weight column in df
    D = []
    Z = 0
    for index, row in df.iterrows():
      #NOTE: I think this is correct but since the error is near 50% alpha is about 0 and updates to weight are minimal
      value = row['weight'] * numpy.exp(-alphas[i] * row[LABEL] * use_stump(stumps[i], row))
      Z += value
      D.append(value)
    
    for index, row in df.iterrows():
      D[index] = D[index]/Z
      assert D[index] > 0

    df['weight'] = D
    
    
  # final is H_final(x) = sgn(sum_t alpha_t h_t(x))
  return alphas, stumps

def bagging(df, attributes, t, sample_size):
  # super confident about this
  C = []
  alphas = []
  for i in range(0, t):
    #  draw m smaples uniformly with replacement
    sample_df = df.sample(n=sample_size, replace=True)
    #learn decision tree c_t with ID3
    C.append(ID3(sample_df, attributes, entropy))
    e_t = calculate_error(C[i], df)
    # print("error is ", e_t)
    alphas.append(alpha_calc(e_t))
  #return votes and Trees
  return alphas, C 

def random_forest(df, attributes, t, sample_size, attribute_sample_size):
  C = []
  alphas = []
  for i in range(0, t):
    # draw m smaples uniformly with replacement
    sample_df = df.sample(n=sample_size, replace=True)
    # learn a tree C_t need to pass a copy
    C.append(ID3(sample_df, attributes.copy(), entropy, is_random=True, random_sample=attribute_sample_size))
    
    #calculate it's vote
    e_t = calculate_error(C[i], df)
    # print(e_t)
    alphas.append(alpha_calc(e_t))
  assert len(alphas) == len(C)
  return  alphas, C

def error_experiment(training_df, attributes, test_df, function):
  bagged_learners = []
  #TODO make use on random and bagged
  for i in range(0, 100):
    print(f"on round {i}")
    sample = training_df.sample(n=1000, replace=False)
    if function == 'bagged':
      alphas_i, trees_i = bagging(sample, attributes, 500, 100)
    elif function == 'random':
      alphas_i, trees_i = random_forest(sample, attributes, 500, 100, 4)
    else:
      print("not a know function")
      return 
    bagged_learners.append((alphas_i, trees_i))
  single_trees = []
  # get first tree in each
  for alphas, trees in bagged_learners:
    single_trees.append(trees[0])
  print("great im  doing fun stats now")
  total_single_variance, total_single_bais = 0, 0
  for index, row in test_df.iterrows():
    row_variance, row_bias = sample_variance_bais_single(row, single_trees)
    total_single_variance += row_variance
    total_single_bais += row_bias
  ave_single_variance = total_single_variance/ len(test_df)
  ave_single_bais = total_single_bais/ len(test_df)

  #add the terms to estimate general squared error
  single_general_squared_error = ave_single_variance + ave_single_bais
  print("now doing stats with bagged")
  total_bagged_variance, total_bagged_bais = 0, 0
  for index, row in test_df.iterrows(): 
    row_variance, row_bias  = sample_variance_bais_bagged(row, bagged_learners)
    total_bagged_variance += row_variance
    total_bagged_bais += row_bias
  ave_bagged_variance = total_bagged_variance/len(test_df)
  ave_bagged_bais = total_bagged_bais/len(test_df)

  bagged_general_squared_error= ave_bagged_bais+ave_bagged_variance

  print("ave_single_variance is: ", ave_single_variance)
  print("ave_single_bais is: ", ave_single_bais)
  print("single_general error is: ", single_general_squared_error)

  print("ave_bagged_vairance is: ", ave_bagged_variance)
  print("ave_bagged_bais is: ", ave_bagged_bais)
  print("bagged error is: ", bagged_general_squared_error)


def sample_variance_bais_bagged(row, bagged_learners):
  predictions = []
  # compute the predictions of the 100 single trees.
  for alphas, trees in bagged_learners:
    # i think
    predictions.append(use_voted_stumps(alphas, trees, row))
  # Take the average, 
  average = 0
  single_tree_baises = []
  for p in predictions:
    average += p
  average = average / len(predictions)
  # subtract the ground-truth label, Real value?????
  ground_truth = row[LABEL]
  # and take square to computethe bias term (see the lecture slides)
  bais = (ground_truth - average) * (ground_truth - average)

  single_tree_baises.append(bais) #TODO remove
  # getting sample variance
  sample_variance = 0
  for p in predictions:
    sample_variance += (average - p) * (average - p)
    assert sample_variance >= 0, f"{sample_variance}, {average}, {p}"
  sample_variance = (1/(len(predictions)-1)) * sample_variance
  assert sample_variance >= 0, f"{sample_variance}, {average}, {p}"
  return sample_variance, bais


def sample_variance_bais_single(row, trees):
  predictions = []
  # compute the predictions of the 100 single trees.
  for tree in trees:
    predictions.append(use_stump(tree, row))
  # Take the average, 
  average = 0
  single_tree_baises = []
  for p in predictions:
    average += p
  average = average / len(predictions)
  # subtract the ground-truth label, Real value?????
  ground_truth = row[LABEL]
  # and take square to compute the bias term (see the lecture slides)
  # bais can be negative but I dont think it is
  bais = (ground_truth-average) * (ground_truth-average)

  single_tree_baises.append(bais)# TODO remove
  # getting sample variance
  sample_variance = 0
  for p in predictions:
    sample_variance += (average - p) * (average - p)
    assert sample_variance >= 0, f"{sample_variance}, {average}, {p}"
  sample_variance = (1/(len(predictions)-1)) * sample_variance
  assert sample_variance >= 0, f"{sample_variance}, {average}, {p}"
  return sample_variance, bais


def model_tests(train_df, test_df, attributes, function, random_sample_size=1):
  # train a bagged of 500
  if function == 'boosted':
    alphas, trees = AdaBoost(train_df, attributes, 500)
  elif function == 'bagged':
    alphas, trees = bagging(train_df, attributes, 500, 100)
  elif function == 'random':
    alphas, trees = random_forest(train_df, attributes, 500, 100, random_sample_size)
  else:
    print("not a known function")
    return
  print("done training")
  # we can now test for every length
  print("i, train_error, test_error")
  test_row_result = {}
  train_row_result = {}
  for i in range(0, 500):
    # a = alphas[:i]
    # t = trees[:i]
    train_correct, train_incorrect = 0, 0
    for index, row in train_df.iterrows():
      if not index in train_row_result.keys():
        train_row_result[index] = 0
      train_row_result[index] += alphas[i] * use_stump(trees[i], row) 
      sgn = 1 if train_row_result[index] > 0 else -1
      if row[LABEL] == sgn:
        train_correct += 1
      else:
        train_incorrect += 1
    # print(f"at iteration {i} correct is {train_correct} incorrect is {train_incorrect}")

    test_correct, test_incorrect = 0, 0
    for index, row in test_df.iterrows():
      if not index in test_row_result.keys():
        test_row_result[index] = 0
      test_row_result[index] += alphas[i] * use_stump(trees[i], row) 
      sgn = 1 if test_row_result[index] > 0 else -1
      if row[LABEL] == sgn:
        test_correct += 1
      else:
        test_incorrect += 1
    # print(f"at iteration {i} correct is {train_correct} incorrect is {train_incorrect}")
    # print(f"at iteration {i} correct is {test_correct} incorrect is {test_incorrect}")
    train_error = train_incorrect/(train_correct  + train_incorrect)
    test_error = test_incorrect/(test_correct  + test_incorrect)
    print(f"{i+1},{train_error},{test_error}")


def get_training_data_with_header(file_path, columns):
  train_df = pandas.read_csv(file_path, names=columns, index_col=0, skiprows=2)
 
  for column in columns:
    #numeric take median
    # and  replace with -  for less then 
    # and replace with  + for greater
    if column.endswith('(num)'):
      median = train_df[column].median()
      train_df[column] = train_df[column].apply(lambda x: "+" if x >= median else '-')
  train_df.index -= 1
  return train_df

  

if __name__ == "__main__":
  if len(sys.argv) == 4:
    #want to just take in train.csv test.csv columns.txt
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    columns_txt = sys.argv[3]
    columns = read_columns(columns_txt)
    LABEL = columns[-1]
    set_label(LABEL)
    train_df = get_training_data(train_csv, columns)
    #train_df =  get_training_data_with_header(sys.argv[1], columns)
    test_df = get_training_data(sys.argv[2], columns)
    rows_num = len(train_df)

    train_df['weight'] = 1/rows_num
    train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 'yes' else -1)
    
    print(train_df)
    
    test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 'yes' else -1)
    print(train_df)
    
    attributes = {}
    for a in columns[:-1]:
      attributes[a] = train_df[a].unique().flatten()

    print("ADAboost")
    model_tests(train_df, test_df, attributes, 'boosted')
    print("bagging")
    
    model_tests(train_df, test_df, attributes, 'bagged')
    print("bagging bias and variance  experiment")
    error_experiment(train_df, attributes, test_df, "bagged")

    print("random forest attribute_sample_size = 2")
    model_tests(train_df, test_df, attributes, 'random', 2)
    print("random forest attribute_sample_size = 4")
    model_tests(train_df, test_df, attributes, 'random', 4)
    print("random forest attribute_sample_size = 6")
    model_tests(train_df, test_df, attributes, 'random', 6)
    print("random forest bias and variance  experiment")
    error_experiment(train_df, attributes, test_df, "random")
  else:

    train_csv = sys.argv[1]
    columns_txt = sys.argv[2]
    columns = read_columns(columns_txt)
    LABEL = columns[-1]
    set_label(LABEL)

    train_df =  get_training_data_with_header(sys.argv[1], columns)
    
    train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
    test_df = train_df.sample(n=6000, replace=False)
    print(test_df)
    train_df = train_df.drop(test_df.index)
    print(train_df)
    test_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    print(test_df)
    print(train_df)

    rows_num = len(train_df)
    train_df['weight'] = 1/rows_num

    attributes = {}
    for a in columns[:-1]:
      attributes[a] = train_df[a].unique().flatten()

    print("single tree")
    single_tree = ID3(train_df, attributes, entropy, 23)
    correct, incorrect = 0, 0
    for index, row in test_df.iterrows():
      if use_stump(single_tree, row) == row[LABEL]:
        correct += 1
      else: 
        incorrect += 1
    print("single tree error is ", incorrect/(correct+ incorrect))

    print("ADAboost")
    model_tests(train_df, test_df, attributes, 'boosted')

    print("bagging")
    model_tests(train_df, test_df, attributes, 'bagged')

    print("random forest sample size = 4")
    model_tests(train_df, test_df, attributes, 'random', 4)
    