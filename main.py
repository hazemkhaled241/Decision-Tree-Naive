# This is a sample Python script.
import numpy as np
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def readForTree(value: float):
    col_names = ['job', 'marital', 'education', 'housing']
    data = pd.read_excel('Bank_dataset.xlsx', names=col_names)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - value))
    classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
    classifier.fit(X_train, Y_train)
    Y_prediction = classifier.predict(X_test)
    print("Decision Tree Accuracy : ", accuracy_score(Y_test, Y_prediction) * 100)


def readForNaive(value: float):
    col_names = ['job', 'marital', 'education', 'housing', 'y']
    data = pd.read_excel('Bank_dataset.xlsx', names=col_names)
    train, test = train_test_split(data, test_size=(1 - value))

    X_test = test.iloc[:, :-1].values
    Y_test = test.iloc[:, -1].values
    yPrediction = naive_bayes_categorical(train, X=X_test, Y="y")
    newPrediction = []
    for x in yPrediction:
        if x == 0:
            newPrediction.append('no')
        else:
            newPrediction.append('yes')

    print("Naive Bayesian Accuracy : ", accuracy_score(Y_test, newPrediction) * 100)


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    df = df[df[Y] == label]
    p_x_given_y = len(df[df[feat_name] == feat_val]) / len(df)
    return p_x_given_y


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i]) / len(df))
    return prior


def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    yPrediction = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        yPrediction.append(np.argmax(post_prob))

    return np.array(yPrediction)


if __name__ == '__main__':
    val = input("Enter percentage of training set: ")
    readForTree(float(val)/100.0)
    readForNaive(float(val)/100.0)
