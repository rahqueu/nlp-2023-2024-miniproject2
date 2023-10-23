import pandas as pd
import sklearn
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# load data and create data frame
train_path = './given_files/train.txt'
df = pd.read_csv(train_path, sep='\t', names=['label', 'review'])

X = df['review']
y = df['label']

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

# applying pre-processing
X = X.apply(apply_preprocessing)

# spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=1)

# Naive Bayes pipeline with CountVectorizer and TF-IDF
nb_pipeline = Pipeline([
    ('vect', CountVectorizer(max_features=5000)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# parameters for grid search
param_grid = {
    'vect__max_features': [1000, 3000, 5000, 10000, 15000, 20000],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [0.1, 0.2, 0.5, 1.0],
}

grid_search = GridSearchCV(nb_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_nb = grid_search.best_estimator_
y_test_pred = best_nb.predict(X_test)

print(f" \n Best parameters: {grid_search.best_params_} \n")

# evaluation
print(f"Testing Set Classification Report: \n {classification_report(y_test, y_test_pred)} \n")

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

labels_accuracies = []
for i in range(len(confusion_matrix)):
    tp = confusion_matrix[i, i]
    fn = confusion_matrix[i, :].sum() - tp
    accuracy = tp / (tp + fn)
    labels_accuracies.append(accuracy)

labels_names = best_nb.classes_
label_accuracy = dict(zip(labels_names, labels_accuracies))
print(f"Label Accuracies: {label_accuracy} \n")

accuracy = accuracy_score(y_test, y_test_pred)*100
print(f"Global Accuracy: {accuracy:.2f}%")
