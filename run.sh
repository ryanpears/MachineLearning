#!
echo "Testing car dataset"
echo "\n"

echo "Testing car/train.csv with information gain"

echo "\n"
for i in 1 2 3 4 5 6
do 
	echo "learning tree with depth $i"
	python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/train.csv ./DecisionTree/car/columns.txt Information_Gain $i False

done

echo "\n"
echo "Testing car/train.csv with majority error"
echo "\n"

for i in 1 2 3 4 5 6
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/train.csv ./DecisionTree/car/columns.txt Majority_Error $i False

done

echo "\n"
echo "Testing car/train.csv with gini index"
echo "\n"

for i in 1 2 3 4 5 6
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/train.csv ./DecisionTree/car/columns.txt Gini_Index $i False

done
echo "\n"
echo "Testing with test data now"
echo "\n"
echo "Testing car/test.csv with information gain"
echo "\n"
for i in 1 2 3 4 5 6
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/test.csv ./DecisionTree/car/columns.txt Information_Gain $i False

done

echo "\n"
echo "Testing car/test.csv with majority error"
echo "\n"
for i in 1 2 3 4 5 6
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/test.csv ./DecisionTree/car/columns.txt Majority_Error $i False

done

echo "\n"
echo "Testing car/test.csv with gini index"
echo "\n"
for i in 1 2 3 4 5 6
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/car/train.csv ./DecisionTree/car/test.csv ./DecisionTree/car/columns.txt Gini_Index $i False

done

echo "\n"

echo " testing bank dataset now"
echo "\n"
echo "unknown treated as a unique value"
echo "\n"
echo "Testing bank/train.csv with information gain"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Information_Gain $i False

done

echo "Testing bank/train.csv with majority error"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Majority_Error $i False

done

echo "Testing bank/train.csv with gini index"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Gini_Index $i False

done


echo "\n"

echo "using test.csv now"

echo "\n"

echo "Testing bank/test.csv with information gain"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Information_Gain $i False

done


echo "\n"

echo "Testing bank/test.csv with majority error"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Majority_Error $i False

done

      
echo "\n"

echo "Testing bank/test.csv with gini index"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Gini_Index $i False

done


echo "\n"

echo "treating unknown values as most common value"
echo "\n"


echo "\n"
echo "testing with training data"
echo "\n"
echo "Testing bank/train.csv with information gain"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Information_Gain $i True

done


echo "\n"
echo "Testing bank/train.csv with majority  error"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"  
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Majority_Error $i True

done

echo "\n"
echo "Testing bank/train.csv with gini index"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"  
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/train.csv ./DecisionTree/bank/columns.txt Gini_Index $i True

done


echo "\n" 
echo "using test.csv  data to test now"

echo "\n"
echo "Testing bank/test.csv with information gain"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"  
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Information_Gain $i True

done


echo "\n"
echo "Testing bank/test.csv with majority  Error"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"  
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Majority_Error $i True

done

echo "\n"
echo "Testing bank/test.csv with gini Index"

echo "\n"
for i in {1..16}
do
        echo "learning tree with depth $i"  
        python3 ./DecisionTree/DecisionTree.py ./DecisionTree/bank/train.csv ./DecisionTree/bank/test.csv ./DecisionTree/bank/columns.txt Gini_Index $i True

done

echo "\n"

echo "should be done with this all now"
