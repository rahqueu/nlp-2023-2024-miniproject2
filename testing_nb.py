import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the data
train_path = './given_files/train.txt'
df = pd.read_csv(train_path, sep='\t', names=['label', 'review'])
print(df.head())
print("\n")

X = df['review']
y = df['label']

# Pre-processing function
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
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've" : "would have",
        "you'd": "you would",
        "you're": "you are",
        "you've": "you have",
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

X = X.apply(apply_preprocessing)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=1)
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)
print("\n")

# Define a Naive Bayes pipeline with CountVectorizer and TF-IDF
text_clf = Pipeline([
    ('vect', CountVectorizer(max_features=5000)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Grid search parameters
param_grid = {
    'vect__max_features': [1000, 3000, 5000, 10000],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0],
}

grid_search = GridSearchCV(text_clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_text_clf = grid_search.best_estimator_

y_test_pred = best_text_clf.predict(X_test)
# Evaluation
print("Testing Set Classification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))
print("\n")

testing_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy: " + str(testing_accuracy))
