#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Embedding Tools
===============
Collection of tools for file and screen printing.

Authors
-------
Sujay Kumar Jauhar <sjauhar@cs.cmu.edu>

Modified by:
Marc Evrard (<marc.evrard@gmail.com>)

License
-------
Copyright 2015 Sujay Kumar Jauhar

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Example
```````
    ./senseretrofit.py -v sample_data/sample_vec.txt.gz -q sample_data/sample_onto.txt.gz
'''
import getopt
import gzip
import sys
from copy import deepcopy

import numpy as np
from scipy.sparse import lil_matrix


help_message = '''
$ python senseretrofit.py -v <vectorsFile> -q <ontologyFile> [-o outputFile] [-n numIters] [-e epsilon] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-q or --ontology to specify path to the ontology (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.sense is used by default)
-n or --numiters to optionally set the number of retrofitting iterations (10 is the default)
-e or --epsilon to optionally set the convergence threshold (0.001 is the default)
-h or --help (this message is displayed)
'''

SENSE_SEP = '%'
VAL_SEP = '#'


def readCommandLineInput(argv):
    ''' Read command line arguments '''
    # specify the possible option switches
    opts, _ = getopt.getopt(argv[1:], "hv:q:o:n:e:", ["help", "vectors=", "ontology=",
                                                        "output=", "numiters=", "epsilon="])
    # default values
    vectorsFile = None
    ontologyFile = None
    outputFile = None
    numIters = 10
    epsilon = 0.001

    setOutput = False
    # option processing
    for option, value in opts:
        if option in ("-h", "--help"):
            print(help_message)
        elif option in ("-v", "--vectors"):
            vectorsFile = value
        elif option in ("-q", "--ontology"):
            ontologyFile = value
        elif option in ("-o", "--output"):
            outputFile = value
            setOutput = True
        elif option in ("-n", "--numiters"):
            numIters = int(value)
        elif option in ("-e", "--epsilon"):
            epsilon = float(value)
        else:
            print(help_message)

    if (vectorsFile == None) or (ontologyFile == None):
        print(help_message)
    else:
        if not setOutput:
            outputFile = vectorsFile + '.sense'
        return (vectorsFile, ontologyFile, outputFile, numIters, epsilon)


def readWordVectors(fpath):
    ''' Read all the word vectors from file.'''
    print("Reading vectors from file...")

    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    wordVectors = {}
    lineNum = 0
    with f_open(fpath) as f:
        vectorDim = int(f.readline().decode().strip().split()[1])
        vectors = np.loadtxt(fpath, dtype=float, comments=None,
                            skiprows=1, usecols=list(range(1, vectorDim + 1)))
        for line in f:
            word = line.decode().lower().strip().split()[0]
            wordVectors[word] = vectors[lineNum]
            lineNum += 1

    print("Finished reading vectors.")

    return wordVectors, vectorDim


def writeWordVectors(wordVectors, vectorDim, fpath):
    ''' Write word vectors to file '''
    print("Writing vectors to file...")

    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    with f_open(fpath, 'w') as f:
        f.write(str(len(wordVectors.keys())) + ' ' + str(vectorDim) + '\n')
        for word in wordVectors:
            f.write(word + ' ' + ' '.join(map(str, wordVectors[word])) + '\n')

    print("Finished writing vectors.")


def addToken2Vocab(token, vocab, vocabIndex, wordVectors):
    ''' Add word sense tokens to a vocabulary relevant to the input vectors.'''
    # check if this sense has a corresponding word vector
    if token.split(SENSE_SEP)[0] not in wordVectors:
        return vocabIndex

    # check if the sense isn't already in the vocabulary
    if token not in vocab:
        vocab[token] = vocabIndex
        return vocabIndex + 1

    return vocabIndex


def readOntology(fpath, wordVectors):
    ''' Read the subset of the ontology relevant to the input vectors.'''

    print("Reading ontology from file...")
    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    # index all the word senses
    vocab = {}
    vocabIndex = 0

    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for token in line:
                # print('**', token)
                token = token.split(VAL_SEP)[0]
                vocabIndex = addToken2Vocab(token, vocab, vocabIndex, wordVectors)
    vocabIndex += 1

    # create the sparse adjacency matrix of weights between senses
    adjacencyMatrix = lil_matrix((vocabIndex, vocabIndex))
    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for i in range(len(line)):
                token = line[i].split(VAL_SEP)
                if token[0] in vocab:
                    # find the row index
                    if i == 0:
                        row = vocab[token[0]]
                    # find the col index of the neighbor and set its weight
                    col = vocab[token[0]]
                    val = float(token[1])
                    adjacencyMatrix[row, col] = val
                else:
                    if i == 0:
                        break
                    continue

    print("Finished reading ontology.")

    # invert the vocab before returning
    vocab = {vocab[k]: k for k in vocab}
    return vocab, adjacencyMatrix.tocoo()


def maxVectorDiff(newVecs, oldVecs):
    ''' Return the maximum differential between old and new vectors
to check for convergence.'''
    maxDiff = 0.0
    for k in newVecs:
        diff = np.linalg.norm(newVecs[k] - oldVecs[k])
        if diff > maxDiff:
            maxDiff = diff
    return maxDiff


def retrofit(wordVectors, vectorDim, senseVocab, ontologyAdjacency, numIters, epsilon):
    ''' Run the retrofitting procedure.'''
    print("Starting the retrofitting procedure...")

    # get the word types in the ontology
    ontologyWords = set([senseVocab[k].split(SENSE_SEP)[0]
                         for k in senseVocab])
    # initialize sense vectors to sense agnostic counterparts
    newSenseVectors = {senseVocab[k]: wordVectors[senseVocab[k].split(SENSE_SEP)[0]]
                       for k in senseVocab}
    # create dummy sense vectors for words that aren't in the ontology (these
    # won't be updated)
    newSenseVectors.update({k + SENSE_SEP + '0:00:00::': wordVectors[k] for k in wordVectors
                            if k not in ontologyWords})

    # create a copy of the sense vectors to check for convergence
    oldSenseVectors = deepcopy(newSenseVectors)

    # run for a maximum number of iterations
    for it in range(numIters):
        newVector = None
        normalizer = None
        prevRow = None
        print('Running retrofitting iter ' + str(it + 1) + '... ')
        # loop through all the non-zero weights in the adjacency matrix
        for row, col, val in zip(ontologyAdjacency.row, ontologyAdjacency.col, ontologyAdjacency.data):
            # a new sense has started
            if row != prevRow:
                if prevRow:
                    newSenseVectors[senseVocab[prevRow]] = newVector / normalizer

                newVector = np.zeros(vectorDim, dtype=float)
                normalizer = 0.0
                prevRow = row

            # add the sense agnostic vector
            if row == col:
                newVector += val * wordVectors[senseVocab[row].split(SENSE_SEP)[0]]
            # add a neighboring vector
            else:
                newVector += val * newSenseVectors[senseVocab[col]]
            normalizer += val

        diffScore = maxVectorDiff(newSenseVectors, oldSenseVectors)
        print('Max vector differential is ' + str(diffScore) + '\n')
        if diffScore <= epsilon:
            break
        oldSenseVectors = deepcopy(newSenseVectors)

    print("Finished running retrofitting.")

    return newSenseVectors


if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    # try opening the specified files
    # try:
    vectors, vectorDim = readWordVectors(commandParse[0])
    senseVocab, ontologyAdjacency = readOntology(commandParse[1], vectors)
    numIters = commandParse[3]
    epsilon = commandParse[4]
    # except:
    #     print("ERROR opening files. One of the paths or formats of the specified files was incorrect.")
    #     sys.exit(2)

    # run retrofitting and write to output file
    writeWordVectors(retrofit(vectors, vectorDim, senseVocab, ontologyAdjacency, numIters, epsilon),
                     vectorDim, commandParse[2])

    print("All done!")
