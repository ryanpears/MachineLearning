# MachineLearning
This is a machine learning library developed by Ryan Pearson forCS5350/6350 in University of Utah

# Decision Tree

use run.sh to run all examples of this.

This can be run as a console app and will print the number of correct, inccorrect and error of provided test data 
from a decision tree learnt from the provided test data. 

Running this app requires pandas and numpy

To run this app use a structure similar to `python3 DecisionTree train.csv test.csv columns.txt Split_Function Max_Depth unknownAsMode`
Arguments to run this as as follows:

train.csv is the training data. No titled columns are allowed. Label column is the last column.

test.csv is the testing data. No titled columns are allowed. Label column is the last column.

columns.txt is where columns for both train.csv and test.csv are defined. include a list of comma seperated strings
to be the column names. The last column will be treated as a label column. end numeric columns names with (num) to 
process them as being less then or greater then the median value of that column.

Split_Function is the function to use to split possible values are "Information_Gain", "Majority_Error" and "Gini_Index"

Max Depth is an integer that will limit the Decision tree to the depth passed in.

UnknownAsMode is a boolean that is used to treat unknown as the mode of that column. if false unknown will be 
treated as a unique value.

# Ensemble learning

use run.sh to test run a script to use these algorithms against both the bank and defualt credit card datasets. 

This can be run as a console app with `python3 Boosting.py train.csv test.csv columns.txt`

This will run the question 2 problems and print out answers in order. 

it can also be run with `python3 Boosting.py data.csv columns.tx` where the dataset will be split randomly into test and training data for question 3

# Linear Regression

use run.sh to test a script to run this on the concrete dataset. 

this file can also be run with `python3 LinearRegression.py train.csv test.csv columns.txt` to print out the answers to question 4

# Perceptron 

use run.sh to run a test script on the bank-note dataset. 

This file can also be run with `python3 Perceptron/Perceptron.py Perceptron/bank-note/train.csv Perceptron/bank-note/test.csv Perceptron/bank-note/columns.txt`
This prints out the answers for question 2

# SVM 

use runHW4.sh to run a test script on the bank-note dataset.

This file can also be run on the bank-note dataset with `python3 SVM/svm.py SVM/bank-note/train.csv SVM/bank-note/test.csv SVM/bank-note/columns.txt`

The syntax for using this script from the command  line is: `python3 SVM/svm.py train.csv test.csv columns.txt` 
where columns.txt is a comma seperated text file of column names.

# Neural Network Classifier

use runHW5.sh to run a test script on the bank-note dataset.

This file can also be run with `python3 NeuralNetworks/neuralnetworkclassifier.py NeuralNetworks/bank-note/train.csv NeuralNetworks/bank-note/test.csv NeuralNetworks/bank-note/columns.txt`

