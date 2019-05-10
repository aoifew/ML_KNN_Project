import pandas as pd
import scipy as sci
from scipy.io import mmread
import numpy as np
import nltk


def cosine_similarity(d1, d2):
    """Computes the cosine similarity between two documents given as arguments
    """
    #Magnitudes for each document vector
    mag_d1 = np.linalg.norm(d1.data)
    mag_d2 = np.linalg.norm(d2.data)
    
    #Calculate dot product of documents
    dotProd = d1.dot(d2.T).data

    #calculate cosine similarity
    cos = dotProd/(mag_d1*mag_d2) 
 
    return cos


def nearest_neighbours(training, test, k, document_no):
    """Computes the k nearest neighbours for a text document with specified document number
    """    
    similarities = [] #List of the similarities from test to each document in the training set
    
    #Iterate over each document in training set
    for j in range(training.shape[0]):
        #Current training doc
        train_doc = training.getrow(j)
            
        #Cosine similarity between test document and current training document
        sim = cosine_similarity(test, train_doc)
            
        #Append training document number and similarity to array
        similarities.append((j+1, sim))
    
        #Sort documents by descending similarity
        similarities = sorted(similarities, key=lambda distance: distance[1], reverse=True)
        
    #Find the k nearest neighbours
    neighbours = []
    for l in range(k):
        neighbours.append(similarities[l])
        
    return neighbours
      
        
def unweighted_knn(training, test, k, document_no, labels):
    """Computes the k nearest neighbours (unweighted) in training set for given test document and 
    labels it based on the result
    """
    #Compute the k nearest neighbours for the test document
    neighbours = nearest_neighbours(training, test, k, document_no)
    
    #Holds label for each of the k nearest neighbours
    votes = [] 
    
    #Adds label from each neighbour to votes list
    for m in range(len(neighbours)):
        index = neighbours[m][0] - 1
        votes.append(labels['label'][index])
    
    #Finds most common vote in list
    count = nltk.FreqDist(votes)
    
    #Return the document number and it's label
    return (document_no, count.max())


def weighted_knn(training, test, k, document_no, labels):
    """Computes the k nearest neighbours (weighted using the inverse distance between the documents) 
    in training set for given test document and labels it based on the result
    """
    #Compute k nearest neighbours
    neighbours = nearest_neighbours(training, test, k, document_no)
    
    #Compute inverse distance to each from similarity
    weights = []
    for i in range(len(neighbours)):
        df_index = neighbours[i][0] - 1
        weights.append((1/(1-neighbours[i][1]), labels['label'][df_index]))
    
    #Array of unique labels in dataset
    l = labels.label.unique()
    
    #Compute weights for each label
    votes = [0]*len(l)

    for w in range(len(weights)):
        for b in range(len(l)):
            if weights[w][1] == l[b]:
                if len(weights[w][0] >=1):
                    votes[b] += weights[w][0][0]

    #return label with total max weight
    index = votes.index(max(votes))
    #Return document number and it's label
    return (document_no, l[index])

def k_nearest_neighbours(training, test, k, weighted, labels):    
    """Computes the k nearest neighbours located in the training set for each document in the test set
    weighted=True computes weighted knn
    """
    classification = [] #Holds document number and label for each document in test set
        
    #Iterate over number of documents in test set
    for g in range(test.shape(0)):
        doc = test.getrow(g) #Current document
            
        #Weighted or unweighted
        if weighted == True:
            classification.append(weighted_knn(training, doc, k, g+1, labels))
                
        else:
            classification.append(unweighted_knn(training, doc, k, g+1, labels))
    
    return(classification)

def accuracy_classifier(c, labels):
    """Calculates the proportion of correctly classified instances by the knn classifier
    """
    #Number of correctly classified instances
    total = 0
    for a in c:
        index = a[0] - 1
        if labels['label'][index] == a[1]:
            total+=1
    
    #% Correctly classified instances
    acc = total/len(c) * 100
    
    return acc

def cross_validation(data, folds, k, weighted, labels):
    """Splits the data into the specified number of folds and performs cross validation using weighted or unweighted knn
    Running time for 10 fold cross validation: 2.5hrs
    """
    print("**************** k =",k,", weighted =",weighted,"****************")
    #Accuracy - %correctly classified instances per fold
    acc = []
    
    #Number of documents
    number = data.shape[0]
    
    #Axis to split data into folds
    fold_split = number//folds
    
    #Create list of document number and fold
    indexes = [0]*number
    
    for i in range(len(indexes)):
        fold = i//fold_split
        if fold > folds:
            fold = folds - 1
        
        indexes[i] = (i+1, fold+1, data.getrow(i))
    
    #Iterate over number of folds    
    for j in range(folds):
        #Testing set = 1 part
        test = [t for t in indexes if t[1] == j+1]
        
        #Training set = all other parts
        training = [t[2] for t in indexes if t[1] != j+1]
        
        #Combine training set back into matrix
        t = sci.sparse.vstack(training)
        t = sci.sparse.csr_matrix(t)
        
        #Remove test labels from labels dataframe
        train_docs = [l[0] for l in indexes if l[1] != j+1] #List of document numbers in training set
        train_labels = labels.loc[labels['document'].isin(train_docs)] 
        train_labels.reset_index(inplace=True,drop=True) #Reset index
        
        print("Fold: ", j+1)
        classification = [] #Holds document number and label for each document in test set
        
        #Iterate over number of documents in test set
        for l in range(len(test)):
            doc = test[l][2] #Current document
            doc_number = test[l][0] #Current document's number
            
            #Weighted or unweighted
            if weighted == True:
                classification.append(weighted_knn(t, doc, k, doc_number, train_labels))
                
            else:
                classification.append(unweighted_knn(t, doc, k, doc_number, train_labels))    

        #Compute accuracy of fold and add to accuracy list
        acc.append(accuracy_classifier(classification, labels))
        print("Correctly Classified Instances in Fold: ", acc[j])

    #Average %correctly classified instances over all folds
    print("\nAvg Correctly Classified Instances: ", np.mean(acc))

    
#read in data
data_input = input("Enter path to data (in .mtx format): ")
matrix = mmread(data_input)

#Convert to csr matrix
data = sci.sparse.csr_matrix(matrix)

#Read in labels for documents in data
label_input = input("Enter path to labels (in .labels format): ")
labels = pd.read_table(label_input, sep=',', names=['document', 'label'])

#Compute accuracy of unweighted knn for values of k [1,10] with 10 fold cross validation
cross_validation(data, 10, 1, False, labels)
#cross_validation(data, 10, 2, False, labels)
#cross_validation(data, 10, 3, False, labels)
#cross_validation(data, 10, 4, False, labels)
#cross_validation(data, 10, 5, False, labels)
#cross_validation(data, 10, 6, False, labels)
#cross_validation(data, 10, 7, False, labels)
#cross_validation(data, 10, 8, False, labels)
#cross_validation(data, 10, 9, False, labels)
#cross_validation(data, 10, 10, False, labels)

#Compute accuracy of weighted knn for values of k [1,10] with 10 fold cross validation
#cross_validation(data, 10, 1, True, labels)
#cross_validation(data, 10, 2, True, labels)
#cross_validation(data, 10, 3, True, labels)
#cross_validation(data, 10, 4, True, labels)
#cross_validation(data, 10, 5, True, labels)
#cross_validation(data, 10, 6, True, labels)
#cross_validation(data, 10, 7, True, labels)
#cross_validation(data, 10, 8, True, labels)
#cross_validation(data, 10, 9, True, labels)
#cross_validation(data, 10, 10, True, labels)

#Leave One Out Cross Validation - Example
#cross_validation(data, 1839, 4, True, labels)