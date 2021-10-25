import sys
import pandas
import numpy

LABEL = ''
FEATURES = []

def standardPerceptron(df, epochs=10):
  weights = [0] * (len(FEATURES)+1)
  r = 0.1 # not sure
  for T in range(0,epochs):
    #shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # learn rows and update on error
    for index, row in df.iterrows():
      train_x = row[:-1]
      train_y = row[LABEL]
      # predict y' = sign(w_t^Transpose x_i)
      prediction = 1 if numpy.dot(weights, train_x) > 0 else -1
      if prediction != train_y:
        #update w_t+1 = w_t + r(y_i x_i)
        x = train_x.to_numpy()
        weights =  weights + r * (train_y * x)
        
  return weights  

def votedPerceptron(df, epochs=10):
  weights = [0] * (len(FEATURES)+1)
  W = []
  c = 1 # maybe zero but likely no
  r = 0.1 # not sure
  for T in range(0,epochs):
    #shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # learn rows and update on error
    for index, row in df.iterrows():
      train_x = row[:-1]
      train_y = row[LABEL]
      # predict y' = sign(w_t^Transpose x_i)
      prediction = 1 if numpy.dot(weights, train_x) > 0 else -1
      if prediction != train_y:
        #update w_t+1 = w_t + r(y_i x_i)
        x = train_x.to_numpy()
        weights =  weights + r * (train_y * x)
        W.append((c, weights))
        c = 1
      else:
        c +=1
        
  return W 


def averagedPerceptron(df, epochs=10):
  weights = [0] * (len(FEATURES)+1)
  W = []
  c = 1 # maybe zero but likely no
  r = 0.1 # not sure
  for T in range(0,epochs):
    #shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # learn rows and update on error
    for index, row in df.iterrows():
      train_x = row[:-1]
      train_y = row[LABEL]
      # predict y' = sign(w_t^Transpose x_i)
      prediction = 1 if numpy.dot(weights, train_x) > 0 else -1
      if prediction != train_y:
        #update w_t+1 = w_t + r(y_i x_i)
        x = train_x.to_numpy()
        weights =  weights + r * (train_y * x)
        W.append((c, weights))
        c = 1
      else:
        c +=1
        
  averaged_weights = [0] * (len(FEATURES)+1)
  total = 0
  for vote, weights in W:
    total += vote
    averaged_weights += vote * weights
  averaged_weights /= total
  
  return  averaged_weights


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
    true_label =  row[LABEL]
    test_data = row[:-1]
    prediction = 1 if numpy.dot(weights, test_data.to_numpy()) > 0 else -1
    if prediction != true_label:
      incorrect += 1
  print("error is ", incorrect/ len(df))


def test_voted_weights(df, voted_weights):
  incorrect = 0
  for index, row in df.iterrows():
    true_label =  row[LABEL]
    test_data = row[:-1]
    
    prediction = 0
    for vote, weights in voted_weights:
      weight_prediction = 1 if numpy.dot(weights, test_data.to_numpy()) > 0 else -1
      prediction += weight_prediction * vote
  
    prediction = 1 if prediction > 0 else -1

    if prediction != true_label:
      incorrect += 1
  print("error is ", incorrect/ len(df))


# def test_averaged_weights(df, voted_weights):
#   incorrect = 0
#   for index, row in df.iterrows():
#     true_label =  row[LABEL]
#     test_data = row[:-1]
    
#     prediction = 0
#     for vote, weights in voted_weights:
#       prediction += vote * numpy.dot(weights, test_data.to_numpy())
#       # prediction += weight_prediction * vote
    
#     prediction = 1 if prediction > 0 else -1

#     if prediction != true_label:
#       incorrect += 1
#   print("error is ", incorrect/ len(df))

if __name__ == "__main__":
  #want argv to be train.csv test.csv columns.txt
  train_csv = sys.argv[1]
  test_csv = sys.argv[2]
  columns_txt = sys.argv[3]
  columns = read_columns(columns_txt)
  LABEL = columns[-1]
  FEATURES = columns[:-1]
  # read csv
  train_df = pandas.read_csv(train_csv, names=columns, index_col=False)
  test_df = pandas.read_csv(test_csv, names=columns, index_col=False)
  # format data
  train_df[LABEL] = train_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  test_df[LABEL] = test_df[LABEL].apply(lambda x: 1 if x == 1 else -1)
  # add bais value to dataframe
  train_df.insert(loc=0, column="baisvalue", value=1)
  test_df.insert(loc=0, column="baisvalue", value=1)
  
  # standard preceptron
  print("running standard perceptron")
  standard_learned_weights = standardPerceptron(train_df)
  print("learned weight vector for standard perceptron is ", standard_learned_weights)
  test_learned_weights(test_df, standard_learned_weights)

  #voted preceptron
  print("running voted preceptron")
  voted_weights = votedPerceptron(train_df)
  for vote, weights in voted_weights:
    print("votes ", vote)
    print("weights ", weights)
  test_voted_weights(test_df, voted_weights)

  #averaged perceptron
  print("running averaged perceptron")
  averaged_weights = averagedPerceptron(train_df)
  print("averaged weights ", averaged_weights)
  # for vote, weights in voted_weights:
  #   print("votes ", vote)
  #   print("weights ", weights)
  test_learned_weights(test_df, averaged_weights)

