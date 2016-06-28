"""
    This is the Main Code to RUN for all the classification Algorithms implemented.
    This code is used to classify tweets based on movie reviews from twitter dataset into 3 different classes
    The classes are: positive, neutral and negative
    The below code uses different datasets, for Training purpose we have used the Rotten Tomatoes Movie Reviews
    In case of Testing we have used Twitter Dataset Reviews
"""

import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import KFold
import argparse
import os

# Variables Initialization
vNegative = []
Negative = []
Positive = []
vPositive = []
data_X = ""
data_Y = ""

"""
    Function to generate the STOPWORDS list
    :parameter NONE
    :returns Stopwords LIST
"""
def generateStopWordList():

    #Fetch the Text File which has all the stopwords from the PATH
    stopWords_dataset = dirPath+"/Data/stopwords.txt"

    #Stopwords List
    stopWords = []

    #Open the stopwords file read the data and store in a list
    try:
        fp = open(stopWords_dataset, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
    except:
        print("ERROR: Opening File")

    return stopWords

"""
    Function to generate Lexicon of sentiments with a Polarity, from a text file AFINN-111.txt
    :parameter Link/PATH of file to get lexicons from
    :returns affin_list
"""
def generateAffinityList(datasetLink):

    affin_dataset = datasetLink
    try:
        affin_list = open(affin_dataset).readlines()
    except:
        print("ERROR: Opening File", affin_dataset)
        exit(0)
    #print(affin_list)

    return affin_list

"""
    This function is used to create a Dictionary of words according to the polarities
    Every word from the AFFIN-111 Lexicon is categorized
    We have taken 4 Categories:
    Very Positive Words, Positive Words, Negative Words, Very Negative Words
    :parameter affin_list
"""
def createDictionaryFromPolarity(affin_list):

    # Create list to store the words and its score i.e. polarity
    words = []
    score = []

    # for every word in AFF-111 list, generate the Words with their scores (polarity)
    for word in affin_list:
        words.append(word.split("\t")[0].lower())
        score.append(int(word.split("\t")[1].split("\n")[0]))

    # print(words)
    # print(score)

    #Categorize words into different Categories
    for elem in range(len(words)):
        if score[elem] == -4 or score[elem] == -5:
            vNegative.append(words[elem])
        elif score[elem] == -3 or score[elem] == -2 or score[elem] == -1:
            Negative.append(words[elem])
        elif score[elem] == 3 or score[elem] == 2 or score[elem] == 1:
            Positive.append(words[elem])
        elif score[elem] == 4 or score[elem] == 5:
            vPositive.append(words[elem])

    # print(vNegative)
    # print(Negative)
    # print(vPositive)
    # print(Positive)

"""
    This function is used for preprocessing the data.
    Here we clean the data, do dimensionaltiy reduction steps
    :parameter Dataset
    :returns processed_data :LIST
"""
def preprocessing(dataSet):

    processed_data = []

    #Make a list of all the Stopwords to be removed
    stopWords = generateStopWordList()

    #For every TWEET in the dataset do,
    for tweet in dataSet:

        temp_tweet = tweet

        #Convert @username to USER_MENTION
        tweet = re.sub('@[^\s]+','USER_MENTION',tweet).lower()
        tweet.replace(temp_tweet, tweet)

        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Replace #HASTAG with only the word by removing the HASH (#) symbol
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)

        tweet.replace(temp_tweet, tweet)

        #Replace all Punctuations
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove additional white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Save the Processed Tweet after data cleansing
        processed_data.append(tweet)

    return processed_data

"""
    This function is used to generate the Feature Vectors for the Training Data,
    and assign a class label to it accordingly
    :parameter Tweet Dataset: dataset, Class Label: type_class
    :returns feature_vector
"""
def FeaturizeTrainingData(dataset, type_class):

    neutral_list = []
    i=0

    # For each Tweet split the Tweet by " " (space) i.e. split every word of the Tweet
    data = [tweet.strip().split(" ") for tweet in dataset]
    #print(data)

    # Feature Vector is to store the feature of the TWEETs
    feature_vector = []

    # for every sentence i.e. TWEET find the words and their category
    for sentence in data:
        # Category count for every Sentence or TWEET
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1
        i+=1

        #Assign Class Label
        if vPositive_count == vNegative_count == Positive_count == Negative_count:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])
            neutral_list.append(i)
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, type_class])

    #print(neutral_list)
    return feature_vector

"""
    This function is used to generate the Feature Vectors for the Test Data
    :parameter Tweet Dataset: dataset
    :returns feature_vector
"""
def FeatureizeTestData(dataset):

    data = [tweet.strip().split(" ") for tweet in dataset]
    #print(data)
    count_Matrix = []
    feature_vector = []

    for sentence in data:
        #print(word)
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1

        if (vPositive_count + Positive_count) > (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "positive"])
            #neutral_list.append(i)
        elif (vPositive_count + Positive_count) < (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "negative"])
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])

        #count_Matrix.append([vPositive_count, Positive_count, Negative_count, vNegative_count])

    return feature_vector

"""
    This function is used to classify the Data using
    Gaussian Naive Bayes Algorithm
    :parameter train_X, train_Y, test_X
    :returns yHat
"""
def classify_naive_bayes(train_X, train_Y, test_X):

    print("Classifying using Gaussian Naive Bayes ...")

    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    return yHat

"""
    This function is used to classify the Data using
    Support Vector Machine Algorithm
    :parameter train_X, train_Y, test_X
    :returns yHat
"""
def classify_svm(train_X, train_Y, test_X):

    print("Classifying using Support Vector Machine ...")

    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)

    return yHat

"""
    This function is used to classify the Data using
    Maximum Entropy Algorithm
    :parameter train_X, train_Y, test_X
    :returns yHat
"""
def classify_maxEnt(train_X, train_Y, test_X):

    print("Classifying using Maximum Entropy ...")
    maxEnt = LogisticRegressionCV()
    maxEnt.fit(train_X, train_Y)
    yHat = maxEnt.predict(test_X)

    return yHat


#########FOR TEST DATA CLASSIFICATION########
def classify_naive_bayes_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Gaussian Naive Bayes ...")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuray: ", Accuracy)
    evaluate_classifier(conf_mat)


def classify_svm_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Support Vector Machine ...")
    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)
    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuracy: ", Accuracy)
    evaluate_classifier(conf_mat)

def classify_maxEnt_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Maximum Entropy ...")
    maxEnt = LogisticRegressionCV()
    maxEnt.fit(train_X, train_Y)
    yHat = maxEnt.predict(test_X)
    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuracy: ", Accuracy)
    evaluate_classifier(conf_mat)

"""
    This function is used to Classify the Tweets from Twitter into its specific class
    Based on the Algorithm to classify
    :parameter file_name

"""
def classify_twitter_data(file_name):

    test_data = open(dirPath+"/Data/"+file_name, encoding="utf8").readlines()
    test_data = preprocessing(test_data)
    test_data = FeatureizeTestData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))

    #Split Data into Features and Classes
    data_X_test = test_data[:,:4].astype(int)
    data_Y_test = test_data[:,4]

    print("Classifying", args.DataSetName)
    #Classify
    if args.Algorithm == "all":
        classify_naive_bayes_twitter(data_X, data_Y, data_X_test, data_Y_test)
        classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test)
        classify_maxEnt_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "gnb":
        classify_naive_bayes_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "svm":
        classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "maxEnt":
        classify_maxEnt_twitter(data_X, data_Y, data_X_test, data_Y_test)

"""
    This function is used to evaluate the performance of the classifier
    It is used to calculate the Precision, Recall, F-Measure and Accuracy
    using the confusion matrix
    :parameter conf_mat Confusion Matrix
"""
def evaluate_classifier(conf_mat):
    Precision = conf_mat[0,0]/(sum(conf_mat[0]))
    Recall = conf_mat[0,0] / (sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)

# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentimental Analysis of Movie Reviews")
    parser.add_argument("DataSetName", help="Dataset to Classify (rottom batvsuper junglebook zootopia deadpool)", metavar='dataset')
    parser.add_argument("Algorithm", help="Classification Algorithm to be used (all gnb svm maxEnt)", metavar='algo')
    parser.add_argument("Crossvalidation", help="Using Cross validation (yes/no)", metavar='CV')
    args = parser.parse_args()
    #print("args are: ",args)

    #fetch th current working dir
    os.chdir('../')        #!!!!!IMPORTANT UNCOMMENT
    dirPath = os.getcwd()
    #print(dirPath)

    # STEP 1: Generate Affinity List
    print("Please wait while we Classify your data ...")
    affin_list = generateAffinityList(dirPath+"/Data/Affin_Data.txt")

    # STEP 2: Create Dictionary based on Polarities from the Lexicons
    createDictionaryFromPolarity(affin_list)

    # STEP 3: Read Data positive and negative Tweets, and do PREPROCESSING
    print("Reading your data ...")
    positive_data = open(dirPath+"/Data/rt-polarity-pos.txt").readlines()
    print("Preprocessing in progress ...")
    positive_data = preprocessing(positive_data)
    #print(positive_data)

    negative_data = open(dirPath+"/Data/rt-polarity-neg.txt").readlines()
    negative_data = preprocessing(negative_data)
    #print(negative_data)

    # STEP 4: Create Feature Vectors and Assign Class Label for Training Data
    print("Generating the Feature Vectors ...")
    positive_sentiment = FeaturizeTrainingData(positive_data, "positive")
    negative_sentiment = FeaturizeTrainingData(negative_data,"negative")
    final_data = positive_sentiment + negative_sentiment
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))

    #Split Data into Features and Classes
    data_X = final_data[:,:4].astype(int)
    data_Y = final_data[:,4]

    # Classifying Entire Dataset
    print("Training the Classifer according to the data provided ...")
    print("Classifying the Test Data ...")
    print("Evaluation Results will be displayed Shortly ...")

    if args.Crossvalidation == "no" or args.Crossvalidation == "No":
        if args.DataSetName == 'rottom':
            if args.Algorithm == "all":
                yHat = classify_naive_bayes(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

                yHat = classify_svm(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

                yHat = classify_maxEnt(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

            elif args.Algorithm == "gnb":
                yHat = classify_naive_bayes(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)
            elif args.Algorithm == "svm":
                yHat = classify_svm(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)
            elif args.Algorithm == "maxEnt":
                yHat = classify_maxEnt(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

        #Classifying Test Data Batman vs Superman
        elif args.DataSetName == "batvsuper":
            classify_twitter_data(file_name="BatmanvSuperman.txt")
        elif args.DataSetName == "junglebook":
            classify_twitter_data(file_name="junglebook.txt")
        elif args.DataSetName == "zootopia":
            classify_twitter_data(file_name="zootopia.txt")
        elif args.DataSetName == "deadpool":
            classify_twitter_data(file_name="deadpool.txt")
        else:
            print("ERROR while specifying Movie Tweets File, please check the name again")

    if args.Crossvalidation == "yes" or args.Crossvalidation == "Yes":
        cv_kFold = KFold(n=len(data_X), n_folds=10, shuffle=True, random_state=5)
        i = 0
        print("Starting "+str(cv_kFold.n_folds)+" Crossvalidation")
        for train_idx, test_idx in cv_kFold:
            X_train, X_test = np.array([data_X[ele] for ele in train_idx]), np.array([data_X[ele] for ele in test_idx])
            Y_train, Y_test = np.array([data_Y[ele] for ele in train_idx]), np.array([data_Y[ele] for ele in test_idx])

            i+=1
            print("Fold: ",i)
            if args.DataSetName == 'rottom':
                if args.Algorithm == "all":
                    yHat = classify_naive_bayes(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                    yHat = classify_svm(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                    yHat = classify_maxEnt(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "gnb":
                    yHat = classify_naive_bayes(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "svm":
                    yHat = classify_svm(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "maxEnt":
                    yHat = classify_maxEnt(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)

            elif args.DataSetName == "batvsuper":
                classify_twitter_data(file_name="BatmanvSuperman.txt")
            elif args.DataSetName == "junglebook":
                classify_twitter_data(file_name="junglebook.txt")
            elif args.DataSetName == "zootopia":
                classify_twitter_data(file_name="zootopia.txt")
            elif args.DataSetName == "deadpool":
                classify_twitter_data(file_name="deadpool.txt")
            else:
                print("ERROR while specifying Movie Tweets File, please check the name again")