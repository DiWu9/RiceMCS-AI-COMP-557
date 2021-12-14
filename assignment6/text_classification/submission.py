"""
Text classification
"""

import util
import operator
from collections import Counter

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        # print("Black List:", self.blackset)
        self.n = n
        self.k = k
        if self.k == -1:
            self.blackset = set(blacklist)
        else:
            self.blackset = set(blacklist[:self.k])
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        if text == "": 
            return 1
        words = text.split(" ")
        existSet = set()
        for word in words:
            if word != "" and word in self.blackset and word not in existSet:
                existSet.add(word)
                if len(existSet) >= self.n:
                    return -1
        return 1
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    featureVector = {}
    words = x.split(" ")
    for word in words:
        featureVector[word] = featureVector.get(word, 0) + 1
    return featureVector
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        featureVector = self.featureFunction(x)
        score = 0
        for word, freq in featureVector.items():
            score += self.params.get(word, 0) * freq
        return score
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    params = {}
    VALUE = (1 , -1)
    for i in range(iters):
        for x, y in trainExamples:
            y = VALUE[labels.index(y)]
            feature = featureExtractor(x)
            score = 0
            for word, freq in feature.items():
                score += params.get(word, 0) * freq
            if score >= 0:
                yh = 1
            else:
                yh = -1
            if y != yh:
                for word, freq in feature.items():
                    params[word] = params.get(word, 0) + freq * y
    return params
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    featureVector = {}
    words = x.split(" ")
    punctuations = {".", "!", "?"}
    for i, word in enumerate(words):
        featureVector[word] = featureVector.get(word, 0) + 1
        if i == 0 or words[i-1] in punctuations:
            featureVector["-BEGIN- " + word] = featureVector.get("-BEGIN- " + word, 0) + 1
        if i < len(words) - 1:
            bigram = word + " " + words[i+1]
            featureVector[bigram] = featureVector.get(bigram, 0) + 1
    return featureVector
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.labels = labels
        self.classifiers = list(classifiers)
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        scores = []
        for label, classifier in self.classifiers:
            score = classifier.classify(x)
            scores.append((label, score))
        return scores

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        scores = self.classify(x)
        ans = max(scores, key=lambda x:x[1])[0]
        return ans
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        scores = []
        for label, classifier in self.classifiers:
            score = classifier.classify(x)
            scores.append((label, score))
        return scores
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    LABEL = ["Yes", "No"]
    learnedClassifiers = []
    for label in labels:
        trainExampleForLabel = []
        for x, y in trainExamples:
            if y == label:
                trainExampleForLabel.append((x, "Yes"))
            else:
                trainExampleForLabel.append((x, "No"))
        param = learnWeightsFromPerceptron(trainExampleForLabel, featureFunction, LABEL, perClassifierIters)
        learnedClassifiers.append((label, WeightedClassifier(LABEL, featureFunction, param)))
    return learnedClassifiers
    # END_YOUR_CODE

'''
if __name__ == "__main__":
    TRAIN_PATH_SPAM = 'data/spam-classification/train'
    TRAIN_PATH_SENTIMENT = 'data/sentiment/train'
    TRAIN_PATH_TOPICS = 'data/topics/train'
    TRAIN_SIZE = 5000
    trainSpamExamples, devSpamExamples = util.holdoutExamples(util.loadExamples(TRAIN_PATH_SPAM)[:TRAIN_SIZE])
    trainDocumentExamples, devDocumentExamples = util.holdoutExamples(util.loadExamples(TRAIN_PATH_TOPICS)[:TRAIN_SIZE])
    blacklist = util.loadBlacklist()
    classifier = RuleBasedClassifier(util.LABELS_SPAM, blacklist)
    classifier.classifyWithLabel(" ".join(blacklist))
    classifier.classifyWithLabel("")
    weights = learnWeightsFromPerceptron(
                trainSpamExamples[:100], 
                extractUnigramFeatures,
                util.LABELS_SPAM,
                3)
    features = {'the': 2, 'over': 1, 'brown': 1, 'lazy': 1, 'fox': 1, 'fence': 1, 'brown fence': 1, 'chased the': 1, 'quick dog': 1, 'fox over': 1, 'chased': 1, 'dog': 1, 'lazy fox': 1, 'The quick': 1, 'the lazy': 1, '-BEGIN- The': 1, 'quick': 1, 'The': 1, 'over the': 1, 'dog chased': 1, 'the brown': 1}
    sentence = "The quick dog chased the lazy fox over the brown fence"
    result = extractBigramFeatures(sentence)
    for k, v in features.items():
        if result.get(k, 0) != v:
            print("Unmatch: " + k, v)
    labels = ["A", "B", "C"]
    weightsA = {'quick': 1, 'lazy' : 1, 'dog' : -1, 'field' : -1 }
    weightsB = {'dog': 1, 'fox' : 1, 'quick' : -1, 'fence' : -1 }
    weightsC = {'dog': -1, 'quick' : -1, 'fence' : 1, 'field' : 1 }
    classifierA = WeightedClassifier( ["A", "!A"], extractUnigramFeatures, weightsA )
    classifierB = WeightedClassifier( ["B", "!B"], extractUnigramFeatures, weightsB)
    classifierC = WeightedClassifier( ["C", "!C"], extractUnigramFeatures, weightsC)
    classifier = OneVsAllClassifier( labels, zip( labels, [classifierA, classifierB, classifierC] ) )
    # print(classifier.classifyWithLabel( "The quick dog was lazy")) # 1, 0, -2
    print(classifier.classifyWithLabel( "The dog was quick unlike the fox")) # 0, 1, -2
    print(classifier.classifyWithLabel( "The dog jumped over the fence and on to the field")) # -2, 0, 1
'''