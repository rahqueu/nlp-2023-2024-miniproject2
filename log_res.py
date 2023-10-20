'''
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

print("LR_TFIDF:\n")
LR_TFIDF()
'''