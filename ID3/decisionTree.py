""" 
 DecisionTree Implementation:
 This is a sample program to give away 
 usage of decision trees(creation and prediction).
 The algorithm models ID3 in the sense that we have not considered other 
 measurements apart from information gain like gini index or gain ratio 
"""
from math import log
from __builtin__ import int, max
from collections import defaultdict
import csv
import random

#Class to encapsulate a decisionNode
class DecisionNode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col # column index of criteria being tested
        self.value=value # value necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.tb=tb # true decision nodes 
        self.fb=fb # false decision nodes

#Sample data to play with when writing your own version of dtree
my_data=[{"referrer":"slashdot","country":"USA","faqclick":"yes","pages":18,"service":"None"},
        {"referrer":"google","country":"France","faqclick":"yes","pages":23,"service":"Premium"},
        {"referrer":"reddit","country":"USA","faqclick":"yes","pages":24,"service":"Basic"},
        {"referrer":"kiwitobes","country":"France","faqclick":"yes","pages":23,"service":"Basic"},
        {"referrer":"google","country":"UK","faqclick":"no","pages":21,"service":"Premium"},
        {"referrer":"(direct)","country":"New Zealand","faqclick":"no","pages":12,"service":"None"},
        {"referrer":"(direct)","country":"UK","faqclick":"no","pages":21,"service":"Basic"},
        {"referrer":"google","country":"USA","faqclick":"no","pages":24,"service":"Premium"},
        {"referrer":"slashdot","country":"France","faqclick":"yes","pages":19,"service":"None"},
        {"referrer":"reddit","country":"USA","faqclick":"no","pages":18,"service":"None"},
        {"referrer":"google","country":"UK","faqclick":"no","pages":18,"service":"None"},
        {"referrer":"kiwitobes","country":"UK","faqclick":"no","pages":19,"service":"None"},
        {"referrer":"reddit","country":"New Zealand","faqclick":"yes","pages":12,"service":"Basic"},
        {"referrer":"slashdot","country":"UK","faqclick":"no","pages":21,"service":"None"},
        {"referrer":"google","country":"UK","faqclick":"yes","pages":18,"service":"Basic"},
        {"referrer":"kiwitobes","country":"France","faqclick":"yes","pages":19,"service":"Basic"}]

"""
Function to get the data from CSV file. 
Assumption made here is that csv file has a header row and 
the dataSet generated should be a list of dictionaries where 
each row is represented by a dictionary 
"""
def getDataFromCSV(fileName):
    dataSet = []
    with open(fileName) as file_obj:
        reader = csv.DictReader(file_obj, delimiter=',')
        ignoredHeader = 0
        for line in reader:
            #We don't realy need the header of the file
            if ignoredHeader == 0:
                pass
            else:
                dataSet.append(line)
            ignoredHeader = 1
    return dataSet

            
"""
Function to getEntropy of the set of rows .
Entropy Formula = Sum over i (-p(i)*log(p(i)))
Here p(i) represents the probability that element will 
belong to ith class if its picked at random from the rows subset.
Example - Suppose there are 10 rows 6 with one labelValue and 4 with another .
    Then entropy = -0.6*log(0.6) - 0.4*log(0.4)
"""
def getEntropy(rows, labelColName):
    #res_dict is the dictionary with the different values for labelColName 
    #as the keys and the count of rows having that value as the values
    res_dict = {}
    total_count = 0
    for row in rows:
        label = row[labelColName]
        label_count = 1
        if label in res_dict.keys():
            label_count = res_dict[label] + 1
        res_dict[label] = label_count
        total_count = total_count + 1
    row_entropy = 0
    for label in res_dict.keys():
        label_count = res_dict[label]
        label_freq = float(label_count)/total_count
        label_entropy = -(label_freq) * (log(label_freq, 2))
        row_entropy = row_entropy + label_entropy
    return row_entropy


"""
Function to partition the rows at a decisionNode.
For numerical data true path is (x >= value)
For categorical data true path is equality .
This will return :
Set1 -> set of rows that would go to true path
Set2 -> set of rows that would go to false path 
"""
def partition(rows, column, value):
    split_func = None
    if isinstance(value, int) or isinstance(value,float):
        split_func = lambda row:row[column] >= value
    else:
        split_func = lambda row:row[column] == value
    set1 = [row for row in rows if split_func(row)]
    set2 = [row for row in rows if not split_func(row)]
    return (set1, set2)


#Function to get the counts of each label in results


"""
Function to return the unique counts of each label in the dataSet
"""
def uniquecounts_dd(rows, labelColName):
    results = defaultdict(lambda: 0)
    for row in rows:
        r = row[labelColName]
        results[r]+=1
    return dict(results) 

"""
Building a tree
1. Iterate through all columns 
2. for each column iterate through all values to find one which has highest entropy gain
3. use that to get the two sets and then for each of them work recursively

This function will recursively create a dcisionTree.
At each call it will check for the column and the value of the column , 
splitting on which will give the maximum information gain 
And then it will recursively call the buildTree method to build the 
true and false subtrees.
labelColName is the name of the column for which we need to make predictions
"""
def buildTree(rows, labelColName):
    columns = rows[0].keys()
    best_gain = 0.0
    best_criteria = None
    best_trueSet = None
    best_falseSet = None
    entropy_rows = getEntropy(rows, labelColName)
    #Iterate over all the columns ignoring the to be predicted column
    for col in columns:
        if col == labelColName:
            continue
        #Get the unique values for the column
        column_values = set([row[col] for row in rows])
        #For all column values divide the dataset into two and 
        #calculate the information gain and update best_gain
        for value in column_values:
            trueSet , falseSet = partition(rows, col, value)
            p = float(len(trueSet)) / len(rows)
            gain = entropy_rows - ((p * getEntropy(trueSet, labelColName)) + ((1-p)*getEntropy(falseSet, labelColName)))
            if gain > best_gain:
                best_gain = gain
                best_trueSet, best_falseSet = trueSet, falseSet
                best_criteria = (col, value)
    if best_gain > 0:
        true_branch = buildTree(best_trueSet, labelColName);
        false_branch = buildTree(best_falseSet, labelColName)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch)
    else:
        return DecisionNode(results=uniquecounts_dd(rows, labelColName));


#Method to print a decisionTree 
def printTree(tree,indent=""):
    # Is this a leaf node?
    if tree.results is not None:
        print str(tree.results)
    else:
        # Print the criteria
        print "Column " + str(tree.col)+" : "+str(tree.value)+"? "

        # Print the branches
        print indent+"True->",
        printTree(tree.tb,indent+"  ")
        print indent+"False->",
        printTree(tree.fb,indent+"  ")

"""
Function to predict the labelValue for the inputRow according 
to the decisionTree provided.
"""
def predictRow(decisionNode, inputRow, labelColName):
    if decisionNode.results is not None:
        result = decisionNode.results
        labelName = ""
        count = 0
        #If its a leaf node then return the labelName with max count
        for label in result.keys():
            if result[label] > count:
                count = result[label]
                labelName = label
        return labelName
    else:
        if isinstance(decisionNode.value, int) or isinstance(decisionNode.value,float):
            decisionVal = inputRow[decisionNode.col]
            if decisionVal >= decisionNode.value:
                return predictRow(decisionNode.tb, inputRow, labelColName)
            else:
                return predictRow(decisionNode.fb, inputRow, labelColName)
        else:
            decisionVal = inputRow[decisionNode.col]
            if decisionVal == decisionNode.value:
                return predictRow(decisionNode.tb, inputRow, labelColName)
            else:
                return predictRow(decisionNode.fb, inputRow, labelColName)

"""
Function to make prediction for testData and return the accuracy of the 
decision tree algorithm
"""
def predictRows(decisionNode, testRows, labelColName):
    predictedLabels = []
    for row in testRows:
        predictedLabels.append(predictRow(decisionNode, row, labelColName))
    return getAccuracy(testRows, predictedLabels, labelColName)

"""
Function to calculate the correctly predicted labels 
"""
def getAccuracy(testRows, predictedLabels, labelColName):
    errorCount = 0
    for index in range(len(testRows)):
        if testRows[index][labelColName] != predictedLabels[index]:
            errorCount = errorCount + 1
    return (1-(float(errorCount)/len(testRows))) * 100

"""
Function to partition the dataSet into train and testData.
testRatio -> ratio of test examples:dataSet size. Default = 0.2
shouldShuffle -> whether the data needs to be shuffled or not. Default = false
"""
def partitionTrainAndTest(dataSet, testRatio = 0.2, shouldShuffle = True):
    dataSize = len(dataSet)
    if shouldShuffle:
        random.shuffle(dataSet)
    trainDataSize = int(dataSize*(1-testRatio))
    trainData = dataSet[:trainDataSize]
    testData = dataSet[trainDataSize:]
    return (trainData,testData)

"""
Function to return the depth of the tree
"""
def findDepth(decisionNode):
    if decisionNode != None:
        return 1 + max(findDepth(decisionNode.tb), findDepth(decisionNode.fb))
    else:
        return 0

"""
Function to count the instances of the different labels for the, to-be predicted
column.
Given a node, it will return the count of T/F instances within the subtree rooted at that node.
We will use this to decide what should be the label if the subtree at a node is pruned
"""
def getCountLabels(decisionNode, labelDict = None):
    if labelDict is None:
        labelDict = {}
    if decisionNode.results is None:
        getCountLabels(decisionNode.tb, labelDict)
        getCountLabels(decisionNode.fb, labelDict)
    else:
        for label in decisionNode.results.keys():
            if label in labelDict.keys():
                labelDict[label] = labelDict[label] + decisionNode.results[label]
            else:
                labelDict[label] = decisionNode.results[label]
    return labelDict
    
"""
Function to prune the tree and check if pruning improves accuracy 
"""
def pruneTree(decisionNode, decisionNodeRoot, currentAccuracy, testDataSet, labelColName):
    #prune the true branch see if it benefits, prune the false branch see if it benefits
    # return the benefited root 
    if decisionNode.results is None:
        labelDict = getCountLabels(decisionNode)
        #Assign the results value of this node with the class label dictionary
        decisionNode.results = labelDict
        #Calculate the accuracy of the tree of this 'pruned' version of tree
        postPruneAccuracy = predictRows(decisionNodeRoot, testDataSet, labelColName)
        if postPruneAccuracy >= currentAccuracy:
            #Actually prune the tree if there is an improve/no change in accuracy
            decisionNode.tb = None
            decisionNode.fb = None
            return decisionNode
        else:
            #If no improvement the return the node to is previous state
            decisionNode.results = None
        #Prune the left and right subtree to see if removing some descendant subtree improves the accuracy
        decisionNodeTrue = pruneTree(decisionNode.tb, decisionNodeRoot, currentAccuracy, testDataSet, labelColName)
        decisionNodeFalse = pruneTree(decisionNode.fb, decisionNodeRoot, currentAccuracy, testDataSet, labelColName)
        #If there was an accuracy improvement then decisionNodeTrue would be different than decisionNode.tb 
        #Hence make that updates(if there was no improvement it would be the same as decisionNode.tb)
        decisionNode.tb = decisionNodeTrue
        decisionNode.fb = decisionNodeFalse
    return decisionNode
     

if __name__ == "__main__":
    """ API Usage Example :
        Using Mushrooms data to use the properties of mushrooms to classify 
        whether the mushroom is edible or poisonous. 
        label column name is 'type' 
    """
    
    #1. Extracting the data from csv as list of dictionaries
    dataSet = getDataFromCSV("../data/mushrooms.csv")
    
    #2. Partitioning dataSet into training and test dataSet
    trainData, testData = partitionTrainAndTest(dataSet)
    #Could use -> trainData, testData = partitionTrainAndTest(dataSet, testRatio = 0.1, shouldShuffle = True)
    
    #3. Building the decision tree using ID3 methodology 
    decisionTree = buildTree(trainData, "type")
    currentAccuracy = predictRows(decisionTree, testData, "type")
    print("Depth of tree before pruning : " + str(findDepth(decisionTree)))
    print("Accuracy of the classifier before pruning : " + str(currentAccuracy))
    
    #print("Tree before pruning --------------")
    #printTree(decisionTree)
    
    #4. Prune the tree with reduced error pruning
    decisionTree = pruneTree(decisionTree, decisionTree, currentAccuracy, testData, "type")
    print("Depth of tree after pruning : " + str(findDepth(decisionTree)))
    print("Accuracy of the classifier after pruning: " + str(predictRows(decisionTree, testData, "type")))
    
    #5. Printing the tree to understand the complexity of the tree 
    print("Tree after pruning --------------")
    printTree(decisionTree)
