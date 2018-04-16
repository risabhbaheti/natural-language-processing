import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util
import time
"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """
        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            
            init_val=0.1
            
            #Initializing weights and biases with random normal
            
            weights_input = tf.get_variable(shape = [Config.hidden_size, Config.n_Tokens*Config.embedding_size], dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean =0.0, stddev = init_val, dtype=tf.float32),
                                            trainable=True, name="weights")
            
            biases_input  =  tf.get_variable(shape = [Config.hidden_size, 1], dtype=tf.float32,
                                            #initializer=tf.random_uniform_initializer(minval=-init_val, maxval=init_val, dtype=tf.float32),
                                            initializer=tf.random_normal_initializer(mean =0.0, stddev = init_val, dtype=tf.float32),
                                            trainable=True, name="bias")
            
            weights_output = tf.get_variable(shape = [parsing_system.numTransitions(), Config.hidden_size], dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean =0.0, stddev = init_val, dtype=tf.float32),
                                            trainable=True, name="weights_out")
            
            #Creating placeholders for the various inputs and lookups
            
            self.train_inputs = tf.placeholder(shape = [Config.batch_size, Config.n_Tokens], dtype=tf.int32, name = "inputs")
            
            self.train_labels = tf.placeholder(shape = [Config.batch_size, parsing_system.numTransitions()] , dtype = tf.float32, name = "labels")
            
            self.test_inputs = tf.placeholder( dtype = tf.int32, name = "test")
            
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            
            embed = tf.reshape(train_embed, [Config.batch_size, -1], name = "embed")
            
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)
            
            #Finding the index for max value in labels to be used later in  cross-entropy loss
            
            label = tf.argmax(self.train_labels, axis=1)
            
            #Calculating the cross-entropy and l2 loss as mentioned in the paper
            
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=self.prediction, labels=label))
            
            l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(embed) + tf.nn.l2_loss(biases_input)
            
            self.loss = cross_entropy + Config.lam*l2
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            
            #Adding softmax for the final output
            self.test_pred = tf.nn.softmax(self.forward_pass(test_embed, weights_input, biases_input, weights_output))

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        #multiplying the weights with the embeddings and adding it to the bias 
        r = tf.add(tf.matmul(embed, weights_input, transpose_b=True), tf.transpose(biases_input))
        
        # Experimenting with relu activation
        #h = tf.nn.relu(r)
        
        # Experimenting with tanh activation
        #h = tf.tanh(r)
        
        # Experimenting with sigmoid activation
        #h = tf.nn.sigmoid(r)
        
        #calculating the power activation 
        h= tf.pow(r,3)
        
        p = tf.matmul(h, weights_output, transpose_b=True)
        
        return p
        

        



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features =[]
    words=[]
    #top three words of stack
    words.append(c.getStack(0))
    words.append(c.getStack(1))
    words.append(c.getStack(2))
    
    #top three words of buffer
    words.append(c.getBuffer(0))
    words.append(c.getBuffer(1))
    words.append(c.getBuffer(2))
    
    #leftmost and rightmost child of stack top
    words.append(c.getLeftChild(words[0],1))
    words.append(c.getLeftChild(words[0],2))
    words.append(c.getRightChild(words[0],1))
    words.append(c.getRightChild(words[0],2))
    words.append(c.getLeftChild(words[1],1))
    words.append(c.getLeftChild(words[1],2))
    words.append(c.getRightChild(words[1],1))
    words.append(c.getRightChild(words[1],2))           
    
    #leftmost of leftmost/rightmost of rightmost
    words.append(c.getLeftChild(words[6],1))
    words.append(c.getLeftChild(words[10],1))
    words.append(c.getRightChild(words[8],1))
    words.append(c.getRightChild(words[12],1))
   
    #Getting word ids for the words
    for i in words:
        features.append(getWordID(c.getWord(i)))
    
    #POS tags for words
    for i in words:
        features.append(getPosID(c.getPOS(i)))
    
    #Labels for last 12 words according to the paper
    for i in words[6:]:
        features.append(getLabelID(c.getLabel(i)))
        
   
        
    return features




def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    count =0
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                #print feat
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
            #if c.tree.equal(trees[i]):
            #    print 'correct!'
            #else:
            #    print 'wrong'
            
                
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    #print knownWords[0:3]
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."
    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

