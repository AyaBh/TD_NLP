from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def make_model(model_type="bayesian"):
    if model_type == "random_forest":
        return RandomForestClassifier()
    elif model_type == "logistic_regression":
        return LogisticRegression()
    elif model_type == "bayesian":
        return MultinomialNB()

