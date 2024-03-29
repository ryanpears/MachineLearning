import sys
import numpy
import pandas
import random

LABEL  = ""
UNKNOWNTREATMENT = False

class DecisionTree:
  def __init__(self, feature):
    self.feature =  feature
    self.children = {}
  
  def add_branch(self, branch, child):
    self.children[branch] = child
  
  def get_branch_keys(self):
    return self.children.keys()

  def __str__(self, level=0):
    ret = "\t"*level+repr(self.feature)+"\n"
    for child, value in self.children.items():
      
      ret +=  str(child) + value.__str__(level+1)
    return ret

def set_label(l):
  global LABEL
  LABEL = l

def ID3(df, Attributes, split_funct, max_depth=None, depth=0, is_random=False, random_sample=1):
  """
  construncts the desicion tree
  S is the set of examples
  Label is the target label ??
  Attributes is the set of measurred attributes
  example_weights are going the be a array or series of  
  weights for each example. not sure how IG will then split.
  """
  if max_depth == None:
    max_depth = len(Attributes)
  # return most common label if all have the same label or max depth is reached 
  if is_unique(df[LABEL]) or (int(depth) >= int(max_depth)) or random_sample > len(Attributes):
    if split_funct == weighted_entropy:
      value_df = df.groupby(LABEL)['weight'].sum()
      value = value_df.idxmax()
    else:
      value = int(df[LABEL].value_counts().idxmax())
    assert (value == -1 or value == 1)
    return DecisionTree(value)
  # 1. Create  Root Node
  # 2. A = Attribute that best splits S
  # NOTE: do steps 2 then 1. 
  gain = float('-inf')
  best_attribute = None
  split_attributes = {}
  if is_random:
   # select a feature subset
    sampled_attributes = random.sample(Attributes.items(), random_sample)
    # turn into a dictionary
    for a in sampled_attributes:
      
      split_attributes[a[0]] = a[1]
      del Attributes[a[0]]
    # print("split attributes is ", split_attributes)
  else:
    split_attributes = Attributes

  for attribute in split_attributes.keys():
    if attribute == LABEL: continue
    poss_gain = information_gain(df, attribute, split_funct)
    assert poss_gain >= -0.01, f"{poss_gain}, {attribute}"
    
    if gain <= poss_gain:
      gain = poss_gain
      best_attribute = attribute
  root = DecisionTree(best_attribute)
  # 3. for each v that A can take
  # print("best attribute", best_attribute)
  all_values = df[best_attribute].unique()
  for value in all_values:
    # a. add new branch to the  tree  A=v
    root.add_branch(value,  None)
    # let  S_v be  subset S where  A = v
    attribute_df = df[df[best_attribute] == value]
      # do the where remove the column
    # c. if S_v is empty  
    if attribute_df.empty:
      # add leaf node with most common value of label
      root.add_branch(value, DecisionTree(df[LABEL].value_counts().idxmax()))
    else:
      #add to subtree
      new_attributes = {key:val for key, val in Attributes.items() if key != best_attribute}
      root.add_branch(value, ID3(attribute_df, new_attributes, split_funct,max_depth, depth+1, is_random, random_sample))
      
   
  #  4. return root
  return root
  
def information_gain(df, attribute, splitFunction):
  total_split_value, total_values = splitFunction(df)
  
  allAttributes= df.groupby(attribute)[attribute].count()
  total_attribute_split = 0

  # print(attribute)
  for index, row in allAttributes.items():
    # print(index)
    attribute_split, count = splitFunction(df.loc[df[attribute] == index])
    # print(f"({count} / {total_values}) * {attribute_split}")
    total_attribute_split += (count / total_values) * attribute_split
    
  # print(total_split_value)
  # print(total_attribute_split)
  # print(total_split_value - total_attribute_split)
  return (total_split_value - total_attribute_split)

def entropy(df):
  """
  returns the entropy of a set
  I think this is ok works on test1
  
  """
  set_entropy = 0
  
  allLabels = df.groupby(LABEL)[LABEL].count()

  total = allLabels.sum()
  
  for index, row in allLabels.items():
    
    probOfLabel = row/total
    set_entropy -= probOfLabel * numpy.log2(probOfLabel)
  
  return set_entropy, total

def weighted_entropy(df):
  """
  entropy but maybe  weighted seriously not  sure wtf I  should be  doing

  """
  #I hinestly feel decently confident about this
  set_entropy = 0
  label_vals = df[LABEL].unique()
  total = df['weight'].sum()
  
  for value in label_vals:
    # this only  works for the first round when all weight sum to 1
    # print(value)
    labelWeightedProb = df.loc[df[LABEL] == value, 'weight'].sum()
    # print(labelWeightedProb)
    assert labelWeightedProb >=0 and labelWeightedProb <=1
    
    set_entropy += (-labelWeightedProb/total) * numpy.log2(labelWeightedProb/total)
  
  assert(set_entropy >= 0)
  return set_entropy, total


def gini_index(df):
  """
  calculates  the gini index
  """
  allLabels = df.groupby(LABEL)[LABEL].count()
  total  = allLabels.sum()
  squares = 0
  for index, row in allLabels.items():
    probOfLabel = row/total
    squares += probOfLabel * probOfLabel
  return (1-squares), total

def majority_error(df):
  """
  calculated the me
  """
  allLabels = df.groupby(LABEL)[LABEL].count()
  total = allLabels.sum()
  max = allLabels.max()
  return (total - max)/total, total


def get_training_data(file_path, columns):
  train_df = pandas.read_csv(file_path, names=columns, index_col=False)
 
  for column in columns:
    #numeric take median
    # and  replace with -  for less then 
    # and replace with  + for greater
    if column.endswith('(num)'):
      median = train_df[column].median()
      train_df[column] = train_df[column].apply(lambda x: "+" if x >= median else '-')
  
  # for unknown as most common value in df
  if UNKNOWNTREATMENT:
    for column in columns:
      mostcommon = train_df[column].value_counts().idxmax()
      train_df[column] = train_df[column].apply(lambda x: mostcommon if x == "unknown"  else x)

  return train_df

def read_columns(file_path):
  with open(file_path, 'r') as file:
    columns = []
    for line in file:
      columnsInLine = line.strip().split(",")
      columns += columnsInLine
    return columns

def test_data(tree, test_file,  columns):
  correct, incorrect = 0, 0
  test_df = pandas.read_csv(test_file, names=columns, index_col=False)
  # also need to do  the median converstion here
  for column in columns:
    #numeric take median
    # and  replace with -  for less then 
    # and replace with  + for greater
    if column.endswith('(num)'):
      median = test_df[column].median()
      test_df[column] = test_df[column].apply(lambda x: "+" if x >= median else '-')

  # for unknown as most common value in df
  if UNKNOWNTREATMENT:
    for column in columns:
      mostcommon = test_df[column].value_counts().idxmax()
      test_df[column] = test_df[column].apply(lambda x: mostcommon if x == "unknown"  else x)

  for index, row in test_df.iterrows():
    if process_row(row, tree):
      correct  +=  1
    else:
      incorrect += 1

  return correct, incorrect

def process_row(row, tree):
  feature = tree.feature

  if tree.children and row[feature] in tree.children.keys():
    return process_row(row, tree.children[row[feature]])
    
  else:
    #leaf node
    return feature == row[LABEL]
  

def is_unique(s):
  a = s.to_numpy() 
  return (a[0] == a).all()


if __name__ == "__main__":
  # argv will likely  look  like Train.csv Test.csv [labels] split_function max depth
  # or probably Train.csv Test.csv [column names] split_function max depth
  # i'd just love some properly formated data but nope.
  # use a * to specify numeric. 
  if len(sys.argv) >= 2:
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    columns = read_columns(sys.argv[3])
    LABEL = columns[-1]
    split_funct_str = sys.argv[4]
    max_depth = sys.argv[5]
    UNKNOWNTREATMENT = sys.argv[6] == 'True'

    split_funct = None
    if split_funct_str == "Information_Gain":
      split_funct = entropy
    elif split_funct_str =="Majority_Error":
      split_funct = majority_error
    elif split_funct_str ==  "Gini_Index":
      split_funct = gini_index
    else:
      print("enter a valid function")
      exit(1)

    train_df =  get_training_data(sys.argv[1], columns)
    
    attributes = {}
    for a in columns[:-1]:
      
      attributes[a] = train_df[a].unique().flatten()
    
    tree = ID3(train_df, attributes, split_funct ,max_depth)
    
    correct,  incorrect = test_data(tree, test_file, columns)
    print("correct is ", correct)
    print("incorrect is ", incorrect)
    error = (incorrect)/(correct + incorrect)
    print("error is ",  error)
  else: 
    print("no data given")