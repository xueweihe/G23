import pandas as pd
import numpy as np
import random
import nltk
import re
import sklearn
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from easybert import Bert
bert = Bert("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")

#Importing datasets#
#Train set
file1 = open("/content/drive/My Drive/Emoji_prediction/us_train.text", "r+", encoding="utf-8")
train_text = file1.readlines()
file2 = open("/content/drive/My Drive/Emoji_prediction/us_train.labels", "r+", encoding="utf-8")
train_labels = file2.readlines()
#Development set
file3 = open("/content/drive/My Drive/Emoji_prediction/us_dev.text", "r+", encoding="utf-8")
dev_text = file3.readlines()
file4 = open("/content/drive/My Drive/Emoji_prediction/us_dev.labels", "r+", encoding="utf-8")
dev_labels = file4.readlines()
#Test set
file5 = open("/content/drive/My Drive/Emoji_prediction/us_test.text", "r+", encoding="utf-8")
test_text = file5.readlines()
file6 = open("/content/drive/My Drive/Emoji_prediction/us_test.labels", "r+", encoding="utf-8")
test_labels = file6.readlines()

###########      Text Preprocessing     #################
#Initalizing lemmatizer and creating a list of stopwords
lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(nltk.corpus.stopwords.words("english"))

#Cleans reviews, removes stopwords and lemmatize to put the reviews in a form that machine can perform better
def clean_reviews(string):
    lemmatized_string = ""
    #Remove HTML information from the string
    cleaner = re.compile('<.*?>')
    processed_string = re.sub(cleaner," ", string)

    #Remove URLS from the string
    processed_string = re.compile(r"https?://[A-Za-z0-9./]+").sub(" ", processed_string)

    #Remove digits and punctuations
    processed_string = re.compile(r"[^a-zA-Z ]").sub(" ", processed_string)

    #Lowercase all the words
    processed_string = processed_string.lower()

    #Does not append stopwords but appends other strings while lemmatizing them
    for word in processed_string.split():
        if word in stopwords:
            continue
        else:
            lemmatized_string += lemmatizer.lemmatize(word) + " "
    return lemmatized_string

#Train, Dev, Test sets of after data preprocessing#
train_processed_text = []
train_processed_labels = []
dev_processed_text = []
dev_processed_labels = []
test_processed_text = []
test_processed_labels = []


for i in train_text:
    train_processed_text.append(clean_reviews(i))
for i in dev_text:
    dev_processed_text.append(clean_reviews(i))
for i in test_text:
    test_processed_text.append(clean_reviews(i))
for i in train_labels:
    train_processed_labels.append(i.replace("\n", ""))
for i in dev_labels:
    dev_processed_labels.append(i.replace("\n", ""))
for i in test_labels:
    test_processed_labels.append(i.replace("\n", ""))

#########     Feature extraction      ##############

train_vector = []
test_vector = []
for i in range(5000):
    train_vector.append(bert.embed(train_processed_text[i]))
for i in range(5000):
    test_vector.append(bert.embed(test_processed_text[i]))

#########     Model Implementation    ##############
#Train random forest model#
rfc = RandomForestClassifier()
rfc.fit(train_vector, train_processed_labels[0:10])
display(rfc.score(train_vector, train_processed_labels[0:10]))

#Predict the test_set
y_predict = rfc.predict(test_vector)
#Calculate precision, recall, f1 score and accuracy of the model
precision = precision_score(test_processed_labels, y_predict, average='macro')
recall = recall_score(test_processed_labels, y_predict, average='macro')
f1 = f1_score(test_processed_labels, y_predict, average='macro')
accuracy = accuracy_score(test_processed_labels, y_predict)

#Print the evaluation measures
print ("Precision: "+str(round(precision,5)))
print ("Recall: "+str(round(recall,5)))
print ("F1-Score: "+str(round(f1,5)))
print ("Accuracy: "+str(round(accuracy,5)))