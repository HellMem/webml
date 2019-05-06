import numpy as np
import pandas as pd
import copy


def get_features_and_labels(data):
    # we separate the labels values from the matrix
    label_values = data[:, 4]
    features = np.delete(data, [4], axis=1)

    # we add a 1's column at the left (for the bias value)
    features = np.insert(features, 0, values=1, axis=1)

    return [features, label_values]


def get_iris_setosa_model(dataset, lr, iters):
    dataset['iris_class'].replace(to_replace=["Iris-setosa"], value=1, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-versicolor"], value=0, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-virginica"], value=0, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(dataset.values)

    [weights, cost_history] = train(features, label_values, weights, lr, iters)

    return weights


def get_iris_versicolor_model(dataset, lr, iters):
    dataset['iris_class'].replace(to_replace=["Iris-setosa"], value=0, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-versicolor"], value=1, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-virginica"], value=0, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(dataset.values)

    [weights, cost_history] = train(features, label_values, weights, lr, iters)

    return weights


def get_iris_virginica_model(data, lr, iters):
    data['iris_class'].replace(to_replace=["Iris-setosa"], value=0, inplace=True)
    data['iris_class'].replace(to_replace=["Iris-versicolor"], value=0, inplace=True)
    data['iris_class'].replace(to_replace=["Iris-virginica"], value=1, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(data.values)

    [weights, cost_history] = train(features, label_values, weights, lr, iters)

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def predict_labels(sepal_length, sepal_width, petal_length, petal_width):
    iris = pd.read_csv("https://s3.amazonaws.com/ml-data-repository-91/iris.csv")

    # learning rate
    lr = 0.01

    # iterations
    iterations = 1500

    iris_setosa_model = get_iris_setosa_model(copy.deepcopy(iris), lr, iterations)  # 0
    iris_versicolor_model = get_iris_versicolor_model(copy.deepcopy(iris), lr, iterations)  # 1
    iris_virginica_model = get_iris_virginica_model(copy.deepcopy(iris), lr, iterations)  # 2

    test_features = np.array([1.0, sepal_length, sepal_width, petal_length, petal_width])
    setosa_pred_value = predict(features=test_features, weights=iris_setosa_model)
    versicolor_pred_value = predict(features=test_features, weights=iris_versicolor_model)
    virginica_pred_value = predict(features=test_features, weights=iris_virginica_model)

    classes = []

    classes.append({'score': setosa_pred_value, 'label': 0})
    classes.append({'score': versicolor_pred_value, 'label': 1})
    classes.append({'score': virginica_pred_value, 'label': 2})

    return classes


def predict(features, weights):
    '''
    Returns 1D array of probabilities
    that the class label == 1
    '''
    z = np.dot(features, weights)
    return sigmoid(z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(Data Size, Number of features)
    Labels: (Data Size, 1)
    Weights:(Number of features,1)
    Returns 1D matrix of predictions
    Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    # Take the error when label=1
    class1_cost = labels * np.log(predictions)

    # Take the error when label=0
    class2_cost = (1 - labels) * np.log(1 - predictions)

    # Take the sum of both costs
    cost = class1_cost + class2_cost

    # Take the average cost
    cost = cost.sum() / observations

    return -cost


def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(Data Size, Number of features)
    Labels: (Data Size, 1)
    Weights:(Number of features, 1)
    '''
    N = len(features)

    # 1 - Get Predictions
    predictions = predict(features, weights)

    # 2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T, predictions - labels)

    # 3 Take the average cost derivative for each feature
    gradient /= N

    # 4 - Multiply the gradient by our learning rate
    gradient *= lr

    # 5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        # Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return weights, cost_history


if __name__ == "__main__":
    classes = predict_labels(6.4, 3.2, 4.5, 1.5)
    #sg_classes = sm.iris_prediction(6.4, 3.2, 4.5, 1.5)
    #print(sg_classes)
    print(classes)
    


'''
if __name__ == "__main__":
    iris = pd.read_csv("https://s3.amazonaws.com/ml-data-repository-91/iris.csv")

    # learning rate
    lr = 0.01

    # iterations
    iterations = 1500

    # we get the features array
    train_data = iris.values

    iris_setosa_model = get_iris_setosa_model(copy.deepcopy(iris), lr, iterations)
    test_features = np.array([1.0, 5.1, 3.5, 1.4, 0.2])
    print(log_reg.predict(features=test_features, weights=iris_setosa_model))
    print('-' * 50)

    iris_versicolor_model = get_iris_versicolor_model(copy.deepcopy(iris), lr, iterations)

    print('-' * 50)
    iris_virginica_model = get_iris_virginica_model(copy.deepcopy(iris), lr, iterations)

    print(log_reg.predict(features=test_features, weights=iris_setosa_model))  # 0
    print(log_reg.predict(features=test_features, weights=iris_versicolor_model))  # 1
    print(log_reg.predict(features=test_features, weights=iris_virginica_model))  # 2

'''
