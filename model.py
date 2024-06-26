from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

def train_model(X_train, y_train):
    """
    Trains a logistic regression model on the training data.

    Parameters:
    X_train (Series): The training text data.
    y_train (Series): The training labels.

    Returns:
    tuple: The trained model and the TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(class_weight='balanced')

    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_tfidf, y_train)

    best_model = grid_search.best_estimator_

    return best_model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Evaluates the model on the test data.

    Parameters:
    model: The trained model.
    vectorizer: The TfidfVectorizer.
    X_test (Series): The test text data.
    y_test (Series): The test labels.

    Returns:
    tuple: The accuracy, classification report, confusion matrix, predicted probabilities, and predicted labels.
    """
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix, y_prob, y_pred

def classify_email(model, vectorizer, email_text):
    """
    Classifies an email as spam or not spam.

    Parameters:
    model: The trained model.
    vectorizer: The TfidfVectorizer.
    email_text (str): The email text to classify.

    Returns:
    str: "Spam" if the email is spam, otherwise "Not Spam".
    """
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"
