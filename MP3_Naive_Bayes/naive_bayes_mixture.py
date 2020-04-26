# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
 


    # TODO: Write your code here
    positive_bigrams, negative_bigrams = dict(), dict()
    positive_unigrams, negative_unigrams = dict(), dict()
    ret = []

    #build bigram dictionaries
    for i in range(len(train_set)):
        current_review = train_set[i]
        current_sentiment = train_labels[i]
        for j in range(len(current_review) - 1):
            current_bigram = current_review[j] + current_review[j+1]
            if current_sentiment == 0:
                negative_bigrams[current_bigram] = negative_bigrams.get(current_bigram, 0) + 1
            elif current_sentiment == 1:
                positive_bigrams[current_bigram] = positive_bigrams.get(current_bigram, 0) + 1

    #build unigram dictionaries                
    for i in range(len(train_set)):
        current_review = train_set[i]
        current_sentiment = train_labels[i]
        for word in current_review:
            if current_sentiment == 0:
                negative_unigrams[word] = negative_unigrams.get(word, 0) + 1
            elif current_sentiment == 1:
                 positive_unigrams[word] = positive_unigrams.get(word, 0) + 1

    total_words_given = get_total_words(train_set)
    total_bigrams_given = get_total_bigrams(train_set)
    total_positive_bigrams, total_negative_bigrams = sum(positive_bigrams.values()), sum(negative_bigrams.values())
    total_positive_unigrams, total_negative_unigrams = sum(positive_unigrams.values()), sum(negative_unigrams.values())

    for review in dev_set:
        unigram_posterior = calculate_unigram_probability(review, positive_unigrams, negative_unigrams, unigram_smoothing_parameter, total_words_given, pos_prior, total_positive_unigrams, total_negative_unigrams)
        bigram_posterior = calculate_bigram_probability(review, positive_bigrams, negative_bigrams, bigram_smoothing_parameter, total_bigrams_given, pos_prior, total_positive_bigrams, total_negative_bigrams)
        mixture_positive_posterior = (1-bigram_lambda)*(unigram_posterior[0]) + (bigram_lambda * bigram_posterior[0])
        mixture_negative_posterior = (1-bigram_lambda)*(unigram_posterior[1]) + (bigram_lambda * bigram_posterior[1])
        ret.append(1) if mixture_positive_posterior >= mixture_negative_posterior else ret.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return ret

# calculates unigram probability given a list of words
def calculate_unigram_probability(review, positive_unigram_words, negative_unigram_words, unigram_smoothing_parameter, total_words_given, pos_prior, total_positive_unigrams, total_negative_unigrams):
    positive_posterior_unigram_temp = math.log10(pos_prior)
    negative_posterior_unigram_temp = 1 - positive_posterior_unigram_temp
    for word in review:
        positive_posterior_unigram_temp += math.log10((positive_unigram_words.get(word, 0) + unigram_smoothing_parameter) / (total_positive_unigrams + unigram_smoothing_parameter * total_words_given))
        negative_posterior_unigram_temp += math.log10((negative_unigram_words.get(word, 0) + unigram_smoothing_parameter) / (total_negative_unigrams + unigram_smoothing_parameter * total_words_given))
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return (positive_posterior_unigram_temp, negative_posterior_unigram_temp)

# calculates bigram probability given list of words
def calculate_bigram_probability(review, positive_bigram_words, negative_bigram_words, bigram_smoothing_parameter, total_bigrams_given, pos_prior, total_positive_bigrams, total_negative_bigrams):
    positive_posterior_bigram_temp = math.log10(pos_prior)
    negative_posterior_bigram_temp = 1 - positive_posterior_bigram_temp
    for j in range(len(review) - 1):
        current_bigram = review[j] + review[j+1]
        positive_posterior_bigram_temp += math.log10((positive_bigram_words.get(current_bigram, 0) + bigram_smoothing_parameter) / (total_positive_bigrams + bigram_smoothing_parameter * total_bigrams_given))
        negative_posterior_bigram_temp += math.log10((negative_bigram_words.get(current_bigram, 0) + bigram_smoothing_parameter) / (total_negative_bigrams + bigram_smoothing_parameter * total_bigrams_given))
    return (positive_posterior_bigram_temp, negative_posterior_bigram_temp)    

# calculates number of unique words given list of lists of words
def get_total_words(words):
    return len(set([word for review in words for word in review]))

# calculates number of unique bigrams given list of lists of words
def get_total_bigrams(words):
    return sum([len(review) - 1 for review in words])