# MachineLearning
This is a machine learning library developed by Ryan Pearson forCS5350/6350 in University of Utah

# Decision Tree

This can be run as a console app and will print the number of correct, inccorrect and error of provided test data 
from a decision tree learnt from the provided test data. 

Running this app requires pandas and numpy

To run this app use a structure similar to `python3 DecisionTree train.csv test.csv columns.txt Split_Function Max_Depth`
Arguments to run this as as follows:

train.csv is the training data. No titled columns are allowed. Label column is the last column.

test.csv is the testing data. No titled columns are allowed. Label column is the last column.

columns.txt is where columns for both train.csv and test.csv are defined. include a list of comma seperated strings
to be the column names. Use "label" as the last column name.

Split_Function is the function to use to split possible values are "Information_Gain", "Majority_Error" and "Gini_Index"

Max Depth is an integer that will limit the Decision tree to the depth passed in.
