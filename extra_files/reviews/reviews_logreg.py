import pandas as pd
import sklearn
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# load training data and create data frame
train_path = 'given_files/train.txt'
train_df = pd.read_csv(train_path, sep='\t', names=['label', 'review'])

# load testing data and create data frame
test_path = './given_files/test_just_reviews.txt'
test_df = pd.read_csv(test_path, sep='\t', header=None, names=['review'])

X_train = train_df['review']
y_train = train_df['label']
X_test = test_df['review']

# pre-processing function
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

    words = lowered.split()
    result = []
    for word in words:
        result.append(contractions.get(word, word))
    preproc = ' '.join(result)

    tokens = word_tokenize(preproc, "english")

    for token in tokens:
        if(all(char in string.punctuation for char in token)):
            tokens.remove(token)

    stop_words = stopwords.words('english')
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return ' '.join(lemmatized_tokens)

# applying pre-processing
X_train = X_train.apply(apply_preprocessing)
X_test = X_test.apply(apply_preprocessing)

# feature extraction: TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features = 5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# model : Logistic Regression
logistic_classifier = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
logistic_classifier.fit(X_train_tfidf, y_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_test_pred = logistic_classifier.predict(X_test_tfidf)

# writing the results to a file
output_path = 'extra_files/results/results_logreg.txt'
with open(output_path, 'w') as output:
    for label in y_test_pred:
        output.write(label + '\n')
print(f"Results written to {output_path}")