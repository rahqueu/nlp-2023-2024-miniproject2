import pandas as pd
import sklearn
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

train_path = './given_files/train.txt'
train_df = pd.read_csv(train_path, sep='\t', names=['label', 'review'])
print(train_df.head())
print("\n")

test_path = './given_files/test_just_reviews.txt'
test_df = pd.read_csv(test_path, sep='\t', header=None, names=['review'])
print(test_df.head())
print("\n")

X_train = train_df['review']
y_train = train_df['label']
X_test = test_df['review']

# pre-processing
def apply_preprocessing(text):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "wasn't": "was not",
        "we'd": "we would",
        "we're": "we are",
        "weren't": "were not",
        "what's": "what is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've" : "would have",
        "you'd": "you would",
        "you're": "you are",
    }
    lowered = text.lower()
    lowered = re.sub(r'[^a-zA-Z\s]', '', lowered)
    words = nltk.word_tokenize(lowered)
    
    result = []
    for word in words:
        result.append(contractions.get(word, word))
    preproc = ' '.join(result)

    tokens = word_tokenize(preproc, "english")
    
    for token in tokens:
        if(all(char in string.punctuation for char in token)):
            tokens.remove(token)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)


for i in range(len(X_train)):
    X_train[i] = apply_preprocessing(X_train[i])

for i in range(len(X_test)):
    X_test[i] = apply_preprocessing(X_test[i])


# feature extraction: tf-idf
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# model : SVC
param_grid = {'C': [0.01, 0.1, 0.575, 1, 2, 4, 10], 'kernel': ['linear', 'rbf']}
svm_classifier = SVC()

grid_search_svm = GridSearchCV(svm_classifier, param_grid=param_grid, cv=3)
grid_search_svm.fit(X_train_tfidf, y_train)
best_svm = grid_search_svm.best_estimator_

X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_test_pred = best_svm.predict(X_test_tfidf)

output_path = 'output/labels_svc.txt'

with open(output_path, 'w') as output:
    for label in y_test_pred:
        output.write(label + '\n')