import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

train_path = './train.txt'

df = pd.read_csv(train_path, sep = '\t', names = ['label', 'review'])
print(df.head())
print("\n")

X = df['review']
y = df['label']

# TODO:
# missing pre-processing step: tokenizaition, lowercasing, tirar stop words, pontuação e outros caracteres random

# dividing data according to train.txt (1400 lines) and test_just_reviews.txt (200 lines) sizing -> 0.125 test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.125, random_state = 1) 
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)
print("\n")

# feature extraction: tf-idf bc its more complex than bag of words and fixes some issues w it but we can also try it
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# model : SVM
def SVM_TFIDF():
    svm_classifier = SVC(kernel = 'linear', C = 0.575)
    svm_classifier.fit(X_train_tfidf, y_train)

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_test_pred = svm_classifier.predict(X_test_tfidf)

    #evaluation
    print("Testing Set Classification Report for Testing Set:")
    print(classification_report(y_test, y_test_pred))
    print("\n")
    
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy: " + str(testing_accuracy))


# model : Logistic Regression
def LR_TFIDF():
    logistic_classifier = LogisticRegression(C = 1.0, penalty= 'l2', solver = 'lbfgs', max_iter = 1000)
    logistic_classifier.fit(X_train_tfidf, y_train)
    
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_test_pred = logistic_classifier.predict(X_test_tfidf)
    
    #evaluation
    print("Testing Set Classification Report for Testing Set:")
    print(classification_report(y_test, y_test_pred))
    print("\n")
    
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy: " + str(testing_accuracy))


print("SVM_TFIDF:\n")
SVM_TFIDF()


print("LR_TFIDF:\n")
LR_TFIDF()


# TODO:
# meter os nomes bem