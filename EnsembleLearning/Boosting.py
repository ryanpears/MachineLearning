import random
import sys

from numpy.core.fromnumeric import prod
from numpy.lib.function_base import average, corrcoef
# TODO use a relative  path
sys.path.insert(0, "/Users/ryanpearson/Documents/CollegeHomework/Fall2021/MachineLearning/MachineLearning/DecisionTree")
import numpy

from DecisionTree import DecisionTree, weighted_entropy, entropy, ID3, get_training_data, read_columns, set_label, process_row

LABEL = 'label'

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
  return 0.5 * numpy.log((1-error)/error)

def AdaBoost(df, attributes, t):
  #TODO
  # for a round t
  alphas = []
  stumps = []
  
  for i in range(0, t):
   
    # make a decsion tree (h_t) of depth 1 
    stumps.append(ID3(df, attributes, weighted_entropy, 1))
    # error_t is the error on the training data
    e_t = calculate_error(stumps[i], df)
    print("error os ", e_t)
    #assert e_t < 0.6
    
    # so my error is the same????
    # calculate its vote  wack alpha_t fuck
    alphas.append(alpha_calc(e_t))
    #this will be updating the weight column in df
    
    # D_t+1(i) = D_t(i) /Z_t (exp(-alpha_t y_i * h_t(x_i)))
    D = []
    Z = 0
    for index, row in df.iterrows():
      alpha = alphas[i]
      stump_result = use_stump(stumps[i], row)
      # print("alpha  is ", alpha)
      # print("stump resulted", stump_result)
      value = numpy.exp(-alphas[i] * row[LABEL] * use_stump(stumps[i], row))
      #value = random.random()* 40
      Z += value
      D.append(value)
    
    for index, row in df.iterrows():
      D[index] = D[index]/Z
      assert D[index] > 0

    df['weight'] = D
    # this repetes in a loop?
    print(df["weight"])
    
  # final is H_final(x) = sgn(sum_t alpha_t h_t(x))
  # really  unsure how to return a  function like this just returning pieces
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

def random_forest(df, attributes, t, sample_size):
  for i in range(0, t):
    # draw m smaples uniformly with replacement
    sample_df = df.sample(n=sample_size, replace=True)
    # learn a 
  return

def random_tree_learn

def experiemnt3(training_df, attributes, test_df):
  bagged_learners = []
  #TODO add extra 0
  for i in range(0, 100):
    print(f"on round {i}")
    sample = training_df.sample(n=1000, replace=False)
    
    alphas_i, trees_i = bagging(sample, attributes, 500, 100)
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

if __name__ == "__main__":
  #want to just take in train.csv test.csv columns.txt
  train_csv = sys.argv[1]
  test_csv = sys.argv[2]
  columns_txt = sys.argv[3]
  columns = read_columns(columns_txt)
  LABEL = columns[-1]
  set_label(LABEL)
  train_df =  get_training_data(sys.argv[1], columns)
  test_df = get_training_data(sys.argv[2], columns)
  rows_num = len(train_df)

  train_df['weight'] = 1/rows_num
  train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 'yes' else -1)

  test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 'yes' else -1)
  print(train_df)
  
  attributes = {}
  for a in columns[:-1]:
    attributes[a] = train_df[a].unique().flatten()
  print("ADAboost")
  # AdaBoost(train_df, attributes, 1)
  print("bagging")
  # bagging(train_df, attributes, 10, 100)
  experiemnt3(train_df, attributes, test_df)
  
  