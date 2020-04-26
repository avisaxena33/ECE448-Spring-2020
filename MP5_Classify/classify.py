# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""
import numpy as np
import heapq
from collections import defaultdict
import math

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights, bias = np.zeros(len(train_set[0])), 0  # initializing feature weights and bias both to 0
    for i in range(max_iter):   # looping through training set of images max_iter amount of times
        for j in range(len(train_set)): # looping through each image in the train set
            current_image = train_set[j]
            current_given_label = train_labels[j]
            activation = np.dot(weights, current_image) + bias
            prediction = 1 if activation > 0 else 0     
            if prediction != current_given_label:   # update weights and bias only if predicted incorrectly
                weights += learning_rate*(current_given_label - prediction)*current_image
                bias += (current_given_label - prediction)*learning_rate
    return weights, bias

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    weights, bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    classified_labels = []
    for i in range(len(dev_set)):   # looping through every image in dev_set
        current_image = dev_set[i]
        activation = np.dot(weights, current_image) + bias
        classified_labels.append(1 if activation > 0 else 0)
    return classified_labels

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    if x >= 0:
        s = (1) / (1 + np.exp(-x))
    else:
        s = np.exp(x) / (1 + np.exp(x))
    return s

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights, bias = np.zeros(len(train_set[0])), 0  # initializing feature weights and bias both to 0
    sample_size = len(train_set)
    for i in range(max_iter):
        gradient = 0
        act_diff = 0
        bias_diff = 0
        for j in range(len(train_set)):
            current_image = train_set[j]
            current_given_label = train_labels[j]
            activation = sigmoid(np.dot(weights, current_image) + bias)
            act_diff += (activation - current_given_label) * current_image
            bias_diff += activation - current_given_label
        weights -= learning_rate * act_diff / sample_size
        bias -= learning_rate * bias_diff / sample_size
    return weights, bias

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    weights, bias = trainLR(train_set, train_labels, learning_rate, max_iter)
    classified_labels = []
    for i in range(len(dev_set)):   # looping through every image in dev_set
        current_image = dev_set[i]
        activation = sigmoid(np.dot(weights, current_image) + bias)
        classified_labels.append(1 if activation >= 0.5 else 0)
    return classified_labels

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    classified_labels = []
    for i in range(len(dev_set)):
        k_closest = []  # max heap
        unclassified_image = np.asarray(dev_set[i])
        euclidean_distance = None   # will be changed
        predicted_label = None  # will be changed
        class_counter = defaultdict()
        for j in range(len(train_set)):
            classified_image = np.asarray(train_set[j])
            euclidean_distance = np.linalg.norm(classified_image - unclassified_image)
            if len(k_closest) < k:
                heapq.heappush(k_closest, (-euclidean_distance, 1 if train_labels[j] else 0))
            elif abs(k_closest[0][0]) > euclidean_distance:
                heapq.heappop(k_closest)
                heapq.heappush(k_closest, (-euclidean_distance, 1 if train_labels[j] else 0))
        for item in k_closest:
            class_counter[item[1]] = class_counter.get(item[1], 0) + 1
        if class_counter.get(0, 0) == class_counter.get(1, 0):
            predicted_label = 0
        elif class_counter.get(0, 0) > class_counter.get(1, 0):
            predicted_label = 0
        else:
            predicted_label = 1
        classified_labels.append(predicted_label)
    return classified_labels
