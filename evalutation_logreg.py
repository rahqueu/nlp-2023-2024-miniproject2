import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

train_path = './train.txt'
train_df = pd.read_csv(train_path, sep='\t', names=['label', 'review'])
print(train_df.head())
print("\n")

test_path = './test_just_reviews.txt'
test_df = pd.read_csv(test_path, sep='\t', header=None, names=['review'])
print(test_df.head())
print("\n")

X_train = train_df['review']
y_train = train_df['label']
X_test = test_df['review']

tfidf_vectorizer = TfidfVectorizer(max_features = 5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_tfidf, y_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_test_pred = logistic_classifier.predict(X_test_tfidf)

output_path = 'test_just_labels_logreg.txt'

with open(output_path, 'w') as output:
    for label in y_test_pred:
        output.write(label + '\n')