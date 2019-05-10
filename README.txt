K-Nearest Neighbours Implementation
Aoife Whelan
15200913

When this script is run, the user is prompted for the file path to the .mtx file representing the text data and the .labels file which contains the associated labels
for this data. The script will then begin to perform cross validation using the supplied data. 

These methods can also be imported to other Python scripts to perform classification.  

This package contains the following methods for computing the k nearest neighbours for text classification.
This implementation supports multi-class classification.

cosine_similarity(d1, d2): 
Input: two 1xn scipy sparse matrices, d1 and d1 representing two documents.  
Computes and returns the cosine similarity between d1 and d2.

nearest_neighbours(training, test, k, document_no):
Input: training, an mxn scipy sparse matrix which represents a number of documents, test, an 1xn matrix representing
a single document, k, the number of nearest neighbours to compute, document_no, the document number of test.
Computes the k most similar documents to the test document in the training set. Returns the document number and 
cosine similarity for the k nearest neighbours in a list.

unweighted_knn(training, test, k, document_no, labels):
Input: training, an mxn scipy sparse matrix which represents a number of documents, test, an 1xn matrix representing
a single document, k, the number of nearest neighbours to compute, document_no, the document number of test, labels,
a pandas dataframe containing the label for each document in the training set.
Computes the k nearest neighbours for test in training and labels test based on the most popular label between the
k neighbours. Returns test's document number and its label.

weighted_knn(training, test, k, document_no, labels)
Input: training, an mxn scipy sparse matrix which represents a number of documents, test, an 1xn matrix representing
a single document, k, the number of nearest neighbours to compute, document_no, the document number of test, labels,
a pandas dataframe containing the label for each document in the training set.
Computes the k nearest neighbours for test in training, weights each neighbour's vote for label based on the inverse distance
between each neighbour and test. Labels test based on the label with the highest weight. Returns test's document number and its label.

k_nearest_neighbours(training, test, k, weighted, labels)
Input: training, an mxn scipy sparse matrix which represents a number of documents, test, an pxn matrix representing
a number of documents, k, the number of nearest neighbours to compute, weighted, a boolean value indicating whether weighted or
un-weighted knn is to be performed, labels, a pandas dataframe containing the label for each document in the training set.
Computes the k nearest neighbours for each document in the test dataset using the appropriate classifier and returns a list of the document
number and the label given by the classifier

accuracy_classifier(c, labels)
Input: c, a list of document numbers and their predicted labels, labels, a dataframe containing the document's actual labels.
Computes the proportion of documents which have been correctly labelled by the classifier. Returns the % accuracy.

cross_validation(data, folds, k, weighted, labels)
Input: data, an mxn scipy sparse matrix which represents a number of documents, folds, the number of folds to perform k-fold cross validation,
weighted, a boolean value to indicate whether the data should be classified by weighted or un-weighted knn, labels, a pandas dataframe of the 
labels in the data.
Splits the data into a number of sections, given by folds, and performs leave one out cross validation on the data. Returns the accuracy of
the classifier on each fold and the average accuracy across all folds.