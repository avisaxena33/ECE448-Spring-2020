"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from collections import defaultdict
import math

# trellis cell class
class node:
    def __init__(self, value, backptr, tag):
        self.value = value
        self.backptr = backptr
        self.tag = tag

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    word_tag_map = defaultdict()
    tags_counter = defaultdict()

    for sentence in train:
        for word, tag in sentence:
            word_tag_map[word] = word_tag_map.get(word, defaultdict())
            word_tag_map[word][tag] = word_tag_map.get(word, defaultdict()).get(tag, 0) + 1
            tags_counter[tag] = tags_counter.get(tag, 0) + 1

    for sentence in test:
        sentence_tags = []
        for word in sentence:
            if word in word_tag_map:
                sentence_tags.append((word, max(word_tag_map[word], key = word_tag_map[word].get)))
            else:
                sentence_tags.append((word, max(tags_counter, key = tags_counter.get)))
        predicts.append(sentence_tags)

    return predicts


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    tags = []
    tags_seen = set()

    for sentence in train:  # store all unique tags for current dataset
        for pair in sentence:
            if pair[1] not in tags_seen:
                tags.append(pair[1])
                tags_seen.add(pair[1])

    laplacian_smoothing_parameter = 0.00001
    predicts = []
    total_train_sentences = len(train)
    tag_pair_counter, tag_first_pair_counter = build_tag_pair_counter(train)
    tag_word_pair_counter, tag_counter = build_tag_word_pair_counter(train)

    initial_tag_probabilities = get_initial_tag_probabilities(train, total_train_sentences, tags, laplacian_smoothing_parameter)
    transition_tag_probabilities = get_transition_tag_probabilities(train, tag_pair_counter, tag_first_pair_counter, tags, laplacian_smoothing_parameter)
    emission_probabilities = get_emission_tag_probabilities(train, tag_word_pair_counter, tag_counter, tags, laplacian_smoothing_parameter)

    for sentence in test:
        trellis = [[node(0, None, '') for j in range(len(tags))] for i in range(len(sentence))]
        for i in range(len(sentence)): # initialize node tags
            for j in range(len(tags)):
                trellis[i][j].tag = tags[j]
        first_word = sentence[0]
        for i in range(len(tags)): # setting up the trellis nodes for the first word in sentence (no back ptr)
            tmp_emis_prob = 0
            if (tags[i], first_word) in emission_probabilities:
                tmp_emis_prob = emission_probabilities[(tags[i], first_word)]
            else:
                tmp_emis_prob = laplacian_smoothing_parameter / (tag_first_pair_counter[tags[i]])
            trellis[0][i].value = math.log10(initial_tag_probabilities[tags[i]]) + math.log10(tmp_emis_prob)
        for i in range(1, len(sentence)): # build the trellis for every word, for every tag, accounting for all prev tags
            current_word = sentence[i]
            for j in range(len(tags)):
                max_previous_tag = (float('-inf'), 0)
                curr_emis_prob = 0
                if (tags[j], current_word) in emission_probabilities:
                    curr_emis_prob = emission_probabilities[(tags[j], current_word)]
                else:
                    curr_emis_prob = laplacian_smoothing_parameter / (tag_first_pair_counter[tags[j]])
                for k in range(len(tags)):
                    tmp_prev_prob = trellis[i-1][k].value + math.log10(transition_tag_probabilities[(tags[k], tags[j])]) + math.log10(curr_emis_prob)
                    if tmp_prev_prob > max_previous_tag[0]:
                        max_previous_tag = (tmp_prev_prob, k)
                trellis[i][j].value = max_previous_tag[0]
                trellis[i][j].backptr = trellis[i-1][max_previous_tag[1]]

        ending_max_prob = float('-inf')
        end = None
        for i in range(len(tags)):
            if trellis[len(sentence)-1][i].value > ending_max_prob:
                ending_max_prob = trellis[len(sentence)-1][i].value
                end = trellis[len(sentence)-1][i]

        curr_sentence_predicts = []
        word_index = len(sentence) - 1
        while end:
            curr_sentence_predicts.append((sentence[word_index], end.tag))
            word_index -= 1
            end = end.backptr
        predicts.append(curr_sentence_predicts[::-1])

    return predicts

# returns a dictionary of count of each tag pair given list of list of (word, tag) tuples
def build_tag_pair_counter(train):
    ret = defaultdict()
    ret2 = defaultdict()
    for sentence in train:
        for i in range(len(sentence) - 1):
            current_tag_pair = (sentence[i][1], sentence[i+1][1])
            ret[current_tag_pair] = ret.get(current_tag_pair, 0) + 1
            ret2[sentence[i][1]] = ret2.get(sentence[i][1], 0) + 1
    return ret, ret2

# returns a dictionary of count of each tag, word pair given list of list of (word, tag) tuples
def build_tag_word_pair_counter(train):
    ret = defaultdict()
    ret2 = defaultdict()
    for sentence in train:
        for word, tag in sentence:
            current_tag_word_pair = (tag, word)
            ret[current_tag_word_pair] = ret.get(current_tag_word_pair, 0) + 1
            ret2[tag] = ret2.get(tag, 0) + 1
    return ret, ret2

# returns a dictionary of probability that each tag is the first word in sentence
def get_initial_tag_probabilities(train, total_train_sentences, tags, laplacian_smoothing_parameter):
    tmp = defaultdict()
    ret = defaultdict()
    for sentence in train:
        tmp[sentence[0][1]] = tmp.get(sentence[0][1], 0) + 1
    for tag in tags:
        ret[tag] = (tmp.get(tag, 0) + laplacian_smoothing_parameter) / (total_train_sentences + (tmp.get(tag, 0) * laplacian_smoothing_parameter))
    return ret

# returns a dictionary of probability that tag(b) follows tag(a)
def get_transition_tag_probabilities(train, tag_pair_counter, tag_first_pair_counter, tags, laplacian_smoothing_parameter):
    ret = defaultdict()
    for i in range(len(tags)):
        for j in range(len(tags)):
            pair = (tags[i], tags[j])
            total_target_pairs = tag_first_pair_counter[pair[0]]
            ret[pair] = (tag_pair_counter.get(pair, 0) + laplacian_smoothing_parameter) / (total_target_pairs + (tag_pair_counter.get(pair, 0) * laplacian_smoothing_parameter))
    return ret

# returns a dictionary of probability that tag(a) yields word(w) (smoothed for only words seen)
def get_emission_tag_probabilities(train, tag_word_pair_counter, tag_counter, tags, laplacian_smoothing_parameter):
    ret = defaultdict()
    for pair, count in tag_word_pair_counter.items():
        target_tag_word_pair = tag_counter[pair[0]]
        ret[pair] = (count + laplacian_smoothing_parameter) / (target_tag_word_pair + (count * laplacian_smoothing_parameter))
    return ret

# returns a dictionary of probability that tag(a) yields word(w) (smoothed for only words seen)
def get_emission_tag_probabilities_v2(train, tag_word_pair_counter, tag_counter, tags, laplacian_smoothing_parameter, hapax_words_tag_prob):
    ret = defaultdict()
    for pair, count in tag_word_pair_counter.items():
        curr_smooth = laplacian_smoothing_parameter * hapax_words_tag_prob[pair[0]]
        target_tag_word_pair = tag_counter[pair[0]]
        ret[pair] = (count + curr_smooth) / (target_tag_word_pair + (count * curr_smooth))
    return ret

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    words_counter = defaultdict()
    hapax_words = set()
    hapax_tag_counter = defaultdict()
    hapax_words_tag_prob = defaultdict() # probability that a tag occurs among set of hapax words

    for sentence in train:
        for pair in sentence:
            words_counter[pair[0]] = words_counter.get(pair[0], 0) + 1

    for word, count in words_counter.items():
        if count == 1:
            hapax_words.add(word)

    for sentence in train:
        for pair in sentence:
            if pair[0] in hapax_words:
                hapax_tag_counter[pair[1]] = hapax_tag_counter.get(pair[1], 0) + 1

    tags = []
    tags_seen = set()
    tag_counter = defaultdict()

    for sentence in train:  # store all unique tags for current dataset
        for pair in sentence:
            tag_counter[pair[1]] = tag_counter.get(pair[1], 0) + 1
            if pair[1] not in tags_seen:
                tags.append(pair[1])
                tags_seen.add(pair[1])

    for tag in tags:
        if tag not in hapax_tag_counter:
            hapax_words_tag_prob[tag] = 1 / len(hapax_words)
        else:
            hapax_words_tag_prob[tag] = hapax_tag_counter[tag] / len(hapax_words)

    laplacian_smoothing_parameter = 0.00001
    predicts = []
    total_train_sentences = len(train)
    tag_pair_counter, tag_first_pair_counter = build_tag_pair_counter(train)
    tag_word_pair_counter, tag_counter = build_tag_word_pair_counter(train)

    initial_tag_probabilities = get_initial_tag_probabilities(train, total_train_sentences, tags, laplacian_smoothing_parameter)
    transition_tag_probabilities = get_transition_tag_probabilities(train, tag_pair_counter, tag_first_pair_counter, tags, laplacian_smoothing_parameter)
    emission_probabilities = get_emission_tag_probabilities_v2(train, tag_word_pair_counter, tag_counter, tags, laplacian_smoothing_parameter, hapax_words_tag_prob)

    for sentence in test:
        trellis = [[node(0, None, '') for j in range(len(tags))] for i in range(len(sentence))]
        for i in range(len(sentence)): # initialize node tags
            for j in range(len(tags)):
                trellis[i][j].tag = tags[j]
        first_word = sentence[0]
        for i in range(len(tags)): # setting up the trellis nodes for the first word in sentence (no back ptr)
            tmp_emis_prob = 0
            if (tags[i], first_word) in emission_probabilities:
                tmp_emis_prob = emission_probabilities[(tags[i], first_word)]
            else:
                tmp_emis_prob = (laplacian_smoothing_parameter * hapax_words_tag_prob[tags[i]]) / (tag_first_pair_counter[tags[i]])
            trellis[0][i].value = math.log10(initial_tag_probabilities[tags[i]]) + math.log10(tmp_emis_prob)
        for i in range(1, len(sentence)): # build the trellis for every word, for every tag, accounting for all prev tags
            current_word = sentence[i]
            for j in range(len(tags)):
                max_previous_tag = (float('-inf'), 0)
                curr_emis_prob = 0
                if (tags[j], current_word) in emission_probabilities:
                    curr_emis_prob = emission_probabilities[(tags[j], current_word)]
                else:
                    curr_emis_prob = (laplacian_smoothing_parameter * hapax_words_tag_prob[tags[j]]) / (tag_first_pair_counter[tags[j]])
                for k in range(len(tags)):
                    tmp_prev_prob = trellis[i-1][k].value + math.log10(transition_tag_probabilities[(tags[k], tags[j])]) + math.log10(curr_emis_prob)
                    if tmp_prev_prob > max_previous_tag[0]:
                        max_previous_tag = (tmp_prev_prob, k)
                trellis[i][j].value = max_previous_tag[0]
                trellis[i][j].backptr = trellis[i-1][max_previous_tag[1]]

        ending_max_prob = float('-inf')
        end = None
        for i in range(len(tags)):
            if trellis[len(sentence)-1][i].value > ending_max_prob:
                ending_max_prob = trellis[len(sentence)-1][i].value
                end = trellis[len(sentence)-1][i]

        curr_sentence_predicts = []
        word_index = len(sentence) - 1
        while end:
            curr_sentence_predicts.append((sentence[word_index], end.tag))
            word_index -= 1
            end = end.backptr
        predicts.append(curr_sentence_predicts[::-1])

    return predicts