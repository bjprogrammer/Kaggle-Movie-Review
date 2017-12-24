'''
    This file contains functions to report on cross-validation in the NLTK.
    The main function to call is cross_validate_evaluate(num_folds, featuresets, label_list)
    Usage:  given previously defined featuresets and list of labels   
        num_folds = 10  # or 5
        label_list = ['pos','neg']
        cross_validate_evaluate(num_folds, featuresets, label_list)
'''
from nltk.metrics import *
import nltk
from nltk.classify import MaxentClassifier

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label
def eval_measures(reflist, testlist, label_list):
    #initialize sets
    # for each label in the label list, make a set of the indexes of the ref and test items
    #   store them in sets for each label, stored in dictionaries
    # first create dictionaries
    ref_sets = {}
    test_sets = {}
    # create empty sets for each label
    for lab in label_list:
        ref_sets[lab] = set()
        test_sets[lab] = set()

    # get gold labels
    for j, label in enumerate(reflist):
        ref_sets[label].add(j)
    # get predicted labels
    for k, label in enumerate(testlist):
        test_sets[label].add(k)

    # lists to return precision and recall for all labels
    precision_list = []
    recall_list = []
    #compute precision and recall for all labels using the NLTK functions
    for lab in label_list:
        precision_list.append ( precision(ref_sets[lab], test_sets[lab]))
        recall_list.append ( recall(ref_sets[lab], test_sets[lab]))

    return (precision_list, recall_list)

# This function computes F-measure (beta = 1) from precision and recall
def Fscore (precision, recall):
    print(precision)
    print(recall)
    if (precision == 0.0) and  (recall == 0.0 ):
      return 0.0
    else:
      return (2.0 * precision * recall) / (precision + recall)

# this function prints precision, recall and F-measure for each label
def print_evaluation(precision_list, recall_list, label_list):
    fscore=[]
    num_folds=0
    num=0
    for index, lab in enumerate(label_list):
        num +=1
        if precision_list[index] is None:
          precision_list[index]=0.0
        if recall_list[index] is None:
          recall_list[index]=0.0
        fscore.append(Fscore(precision_list[index],recall_list[index]))
        if fscore[num_folds]==0:
          num-=1
        num_folds += 1
    print('average precision', sum(precision_list)/num_folds)
    print('average recall   ', sum(recall_list)/num_folds)
    print('F-score  ',sum(fscore)/num)

# This function performs the cross-validation, creating classifier models for each fold
#    In each fold, it also applies the model to the reference set, getting a list of predicted labels
#    The resulting final list collects all the reference/gold labels and test/predicted labels
def naive_bayes(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("Naive Bayes Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")

def GIS(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("GIS Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = MaxentClassifier.train(train_this_round, 'GIS', max_iter = 1)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")

def IIS(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("IIS Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = MaxentClassifier.train(train_this_round, 'IIS', max_iter = 1)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")

def multinomialNB(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("MultinomialNB Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(MultinomialNB())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")	

def BernouliNB(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("BernouliNB Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(BernoulliNB())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def decisiontree(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("Decision Tree Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(DecisionTreeClassifier())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def logisticregression(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("Logistic Regression Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(LogisticRegression())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def sgdc(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("SGDC Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(SGDClassifier())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def svc(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list =[] 
    print("SVC Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(SVC())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    
    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def linearsvc(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("Linear SVC Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(LinearSVC())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    
    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def nusvc(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("NuSVC Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(NuSVC(nu=0.01))
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    
    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")
	
def randomforests(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    accuracy_list = []
    print("Random Forests Classifier")
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = SklearnClassifier(RandomForestClassifier())
        classifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    
    print('Done with cross-validation')
    # call the evaluation measures function    
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
    print(" ")


