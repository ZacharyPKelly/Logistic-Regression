#######################################################################################################################################
# This file contains code in order to implement the logistic regression model with sigmoid activation function from scratch as outlined
# in Lab 5 of AUCSC 460.
#
# Class: AUCSC 460
# Name: Zachary Kelly
# Student ID: 1236421
# Date: March 20th, 2024
#######################################################################################################################################

# IMPORTS #

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# METHODS #

def accuracy(yTrue, yPredicted, yPredictedUnrounded):

    """
    Takes the actual values of y, the predicted and rounded values of y, and the unrounded predicted values of y
    and prints them side by side followed by the accuarcy of the model

    ytrue               : An array containing the true values of y
    yPredicted          : An array containing the rounded predicted values of y made by the model
    yPredictedUnrounded : An array containing the unrounded predicted values of y made by the model
    """

    print()

    print("Actual | Pred Rounded | Pred Unrounded")
    print("--------------------------------------")

    #Print the acutal value of y, the rounded predicted value of y, and the unrounded predicted value of y
    for i in range(len(yTrue)):

        print("{}      | {}          | {}".format(yTrue[i], yPredicted[i], yPredictedUnrounded[i]))

    #Print the accuracy of our model
    print()
    print("Accuracy: {}%".format(accuracy_score(yTrue, yPredicted) * 100))

def backwardPass(lossDerivative, predProbDerivative, x1, x2, weightArray, learningRate):

    """
    Calculates the gradient of loss w.r.t each weight and applies said gradient of loss
    to its associated weight.

    lossDerivative     : The derivative of the loss
    predProbDerivative : The derivative of the predicted probability yHat w.r.t the input of the sigmoid function
    x1                 : Feature one of the given example w.r.t the predicted probability yHat
    x2                 : Feature two of the given example
    weightArray        : The array containing the weights
    learningRate       : The choosen learning rate for the model

    Returns            : An array containg the three new values of the weights
    """

    #Calculate the gradient of loss w.r.t each weight
    w0GradientOfLoss = lossDerivative * predProbDerivative * 1
    w1GradientOfLoss = lossDerivative * predProbDerivative * x1
    w2GradientOfLoss = lossDerivative * predProbDerivative * x2

    #Calculate new weights
    w0Value = weightArray[0] - ( learningRate * w0GradientOfLoss)
    w1Value = weightArray[1] - ( learningRate * w1GradientOfLoss)
    w2Value = weightArray[2] - ( learningRate * w2GradientOfLoss)

    return [w0Value, w1Value, w2Value]

def binaryCrossEntropyFunction(y, yHat):

    """
    Calculates the loss of the given prediction (yHat) compared to the actual value of y

    y       : The actual value of y
    yHat    : The predicted value of y made by the model

    Returns : The value of the loss as calculated by binary cross entropy
    """

    #Calculate the binary cross entropy loss
    lossValue = -( ( y * np.log(yHat) ) + ( 1 - y ) * (np.log( 1 - yHat) ) )

    return lossValue

def derivativeOfLoss(y, yHat):

    """
    Calculates the derivate of the loss w.r.t the predicted probability yHat

    y       : The actual value of y
    yHat    : The predicted value of y made by the model

    Returns : The value of the derivative of the loss w.r.t the predicted probability yHat
    """

    #Calculate the derivate of the loss w.r.t the predicted probability yHat
    derivativeLossValue = -1 * ( ( y / yHat ) - ( ( 1 - y ) / (1 - yHat ) ) )

    return derivativeLossValue

def derivativeOfPredictedProb(yHat):

    """
    Calculates the derivative of the predicted probability (yHat) w.r.t the input of the sigmoid function

    yHat : The predicted value of y made by the model

    Returns: The value of the derivative of the predicted probability (yHat) w.r.t the input of the sigmoid function
    """

    #Calculate the derivative of the predicted probability (yHat) w.r.t the input of the sigmoid function
    derivativePredProbValue = yHat * (1 - yHat)

    return derivativePredProbValue

def initializeWeights():

    """
    Initializes the weights for the model as floats between -1.0 and 1.0

    Returns : An array containing the three weights for the model
    """

    #Create array containing three random weights between -1.0 and 1.0
    weightArray = np.random.uniform(size = 3, low = -1.0, high = 1.0)

    return weightArray

def predict(testingSetX, testingSetY, weightSet):

    """
    Makes a prediction using the model for every example in the testingSetX then compares the results
    to the data in testingSetY in order to calculate the accuracy of the model.

    testingSetX : An array containing all of our testing X data to make predictions with
    testingSetY : An array containing all of our testing Y data to compare predictions with
    weightSet   : The refined weights of the model after having been trained by the model
    """

    #Initialize arrays to hold our rounded and unrounded predictions
    predictedValues = []
    predictedValuesUnrounded = []

    #Iterate through all the testing examples
    for i in range(len(testingSetX)):

        #Get our features from our given example
        x1 = testingSetX[i][0]
        x2 = testingSetX[i][1]

        #Calculate the weighted sum of our weights and features from the given example
        weightedSum = weightSet[0] + ( x1 * weightSet[1] ) + ( x2 * weightSet[2] )

        #Make our prediction based on the weighted sum
        yHat = sigmoid(weightedSum)

        #Append the rounded and unrounded prediction to the associated array
        predictedValuesUnrounded.append(yHat)
        predictedValues.append(np.round(yHat))           

    #Print the results and accuracy of our model on the testing set
    accuracy(testingSetY, predictedValues, predictedValuesUnrounded)

def sigmoid(z):

    """
    Calculates the predicted probability for a given example as a range between 1 or 0.

    z : The weighted sum of the given example and the weights

    Returns : The predicted probability for a given example as a range between 1 or 0
    """

    #Calculate the predicted probability
    sigmoidValue = 1 / ( 1 + np.exp(-z) )

    return sigmoidValue

# DRIVER CODE #

def main():

    #Set random seed; set to int for consistent random array, leave empty for a new array each time the program is run
    np.random.seed(0)

    #Initialize and structure our data arrays.
    num_samples_per_class = 100
    x1 = np.random.randn(num_samples_per_class, 2) + np.array([2, 2])
    x2 = np.random.randn(num_samples_per_class, 2) + np.array([-2, -2])
    x = np.vstack([x1, x2])
    y = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    #Randomly permute our arrays.
    shuffle_idx = np.random.permutation(len(x))
    x = x[shuffle_idx]
    y = y[shuffle_idx]

    #Split our data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

    #Initialize our weight array
    weights = initializeWeights()

    #Initialize and set our learning rate
    learningRate = 0.01

    #Initialize and set the number of iterations our model will go through
    numberOfIterations = 1000

    for i in range(numberOfIterations):

        #adjust weights based on the predicted output for each example in training set
        for j in range(len(xTrain)):

            #Get our features from our given example
            x1 = xTrain[j][0]
            x2 = xTrain[j][1]

            #Calculate the weighted sum of our weights and features from the given example
            weightedSum = weights[0] + ( x1 * weights[1] ) + ( x2 * weights[2] )

            #Make our prediction based on the weighted sum
            yHat = sigmoid(weightedSum)

            #Calculate the loss and print it out to the screen
            # print("LOSS :", binaryCrossEntropyFunction(yTrain[j], yHat))

            #Calculate the derivative of the loss w.r.t the predicted probability yHat
            yHatDerivativeOfLoss = derivativeOfLoss(yTrain[j], yHat)

            #Calculate the derivative of the predicted probabilty yHat w.r.t the input of the sigmoid function
            yHatDerivativeOfPredProb = derivativeOfPredictedProb(yHat)

            #Calculate the values of the new weights
            weights = backwardPass(yHatDerivativeOfLoss, yHatDerivativeOfPredProb, x1, x2, weights, learningRate)

    #Make predications on testing set and print accuracy
    predict(xTest, yTest, weights)

if __name__ == "__main__":

    main()