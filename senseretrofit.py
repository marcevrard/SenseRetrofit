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
$ python senseretrofit.py -v <emb_in_fpath> -q <onto_fpath> [-o emb_out_fpath] [-n n_iters] [-e epsilon] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-q or --ontology to specify path to the ontology (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<emb_in_fpath>.sense is used by default)
-n or --numiters to optionally set the number of retrofitting iterations (10 is the default)
-e or --epsilon to optionally set the convergence threshold (0.001 is the default)
-h or --help (this message is displayed)
'''

SENSE_SEP = '%'
VAL_SEP = '#'


def read_args(argv):
    ''' Read command line arguments '''
    # specify the possible option switches
    opts, _ = getopt.getopt(argv[1:], "hv:q:o:n:e:",
                            ["help", "vectors=", "ontology=", "output=", "numiters=", "epsilon="])
    # default values
    emb_in_fpath = None
    onto_fpath = None
    emb_out_fpath = None
    n_iters = 10
    epsilon = 0.001

    set_out_fpath = False
    # option processing
    for option, value in opts:
        if option in ("-h", "--help"):
            print(help_message)
        elif option in ("-v", "--vectors"):
            emb_in_fpath = value
        elif option in ("-q", "--ontology"):
            onto_fpath = value
        elif option in ("-o", "--output"):
            emb_out_fpath = value
            set_out_fpath = True
        elif option in ("-n", "--numiters"):
            n_iters = int(value)
        elif option in ("-e", "--epsilon"):
            epsilon = float(value)
        else:
            print(help_message)

    if (emb_in_fpath is None) or (onto_fpath is None):
        print(help_message)
    else:
        if not set_out_fpath:
            emb_out_fpath = emb_in_fpath + '.sense'
        return (emb_in_fpath, onto_fpath, emb_out_fpath, n_iters, epsilon)


def read_emb_in(fpath):
    ''' Read all the word vectors from file.'''
    print("Reading vectors from file...")

    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    wvecs = {}
    l_idx = 0
    with f_open(fpath) as f:
        v_dim = int(f.readline().decode().strip().split()[1])
        vectors = np.loadtxt(fpath, dtype=float, comments=None,
                             skiprows=1, usecols=list(range(1, v_dim + 1)))
        for line in f:
            word = line.decode().lower().strip().split()[0]
            wvecs[word] = vectors[l_idx]
            l_idx += 1

    print("Finished reading vectors.")

    return wvecs, v_dim


def write_emb_out(wvecs, v_dim, fpath):
    ''' Write word vectors to file '''
    print("Writing vectors to file...")

    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    with f_open(fpath, 'w') as f:
        f.write(str(len(wvecs.keys())) + ' ' + str(v_dim) + '\n')
        for word in wvecs:
            f.write(word + ' ' + ' '.join(map(str, wvecs[word])) + '\n')

    print("Finished writing vectors.")


def add_token2voc(token, vocab, voc_idx, wvecs):
    ''' Add word sense tokens to a vocabulary relevant to the input vectors.'''
    # check if this sense has a corresponding word vector
    if token.split(SENSE_SEP)[0] not in wvecs:
        return voc_idx

    # check if the sense isn't already in the vocabulary
    if token not in vocab:
        vocab[token] = voc_idx
        return voc_idx + 1

    return voc_idx


def read_ontology(fpath, wvecs):
    ''' Read the subset of the ontology relevant to the input vectors.'''

    print("Reading ontology from file...")
    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    # index all the word senses
    vocab = {}
    voc_idx = 0

    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for token in line:
                # print('**', token)
                token = token.split(VAL_SEP)[0]
                voc_idx = add_token2voc(token, vocab, voc_idx, wvecs)
    voc_idx += 1

    # create the sparse adjacency matrix of weights between senses
    adjacency_mtx = lil_matrix((voc_idx, voc_idx))
    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for idx, el in enumerate(line):
                token = el.split(VAL_SEP)
                if token[0] in vocab:
                    # find the row index
                    if idx == 0:
                        row = vocab[token[0]]
                    # find the col index of the neighbor and set its weight
                    col = vocab[token[0]]
                    val = float(token[1])
                    adjacency_mtx[row, col] = val
                else:
                    if idx == 0:
                        break
                    continue

    print("Finished reading ontology.")

    # invert the vocab before returning
    vocab = {vocab[k]: k for k in vocab}
    return vocab, adjacency_mtx.tocoo()


def max_vec_diff(new_vecs, old_vecs):
    ''' Return the maximum differential between old and new vectors
    to check for convergence.'''
    max_diff = 0.0
    for k in new_vecs:
        diff = np.linalg.norm(new_vecs[k] - old_vecs[k])
        if diff > max_diff:
            max_diff = diff
    return max_diff


def retrofit(wvecs, v_dim, sense_voc, onto_adjacency, n_iters, epsilon):
    ''' Run the retrofitting procedure.'''
    print("Starting the retrofitting procedure...")

    # get the word types in the ontology
    onto_words = set([sense_voc[k].split(SENSE_SEP)[0]
                      for k in sense_voc])
    # initialize sense vectors to sense agnostic counterparts
    new_sense_vecs = {sense_voc[k]: wvecs[sense_voc[k].split(SENSE_SEP)[0]]
                      for k in sense_voc}
    # create dummy sense vectors for words that aren't in the ontology (these
    # won't be updated)
    new_sense_vecs.update({k + SENSE_SEP + '0:00:00::': wvecs[k] for k in wvecs
                           if k not in onto_words})

    # create a copy of the sense vectors to check for convergence
    old_sense_vecs = deepcopy(new_sense_vecs)

    # run for a maximum number of iterations
    for itr in range(n_iters):
        new_vec = None
        normalizer = None
        prev_row = None
        print("Running retrofitting iter {:2d}...".format(itr + 1), end=' ')
        # loop through all the non-zero weights in the adjacency matrix
        for row, col, val in zip(onto_adjacency.row, onto_adjacency.col, onto_adjacency.data):
            # a new sense has started
            if row != prev_row:
                if prev_row:
                    new_sense_vecs[sense_voc[prev_row]] = new_vec / normalizer

                new_vec = np.zeros(v_dim, dtype=float)
                normalizer = 0.0
                prev_row = row

            # add the sense agnostic vector
            if row == col:
                new_vec += val * wvecs[sense_voc[row].split(SENSE_SEP)[0]]
            # add a neighboring vector
            else:
                new_vec += val * new_sense_vecs[sense_voc[col]]
            normalizer += val

        diff_score = max_vec_diff(new_sense_vecs, old_sense_vecs)
        print("Max vector differential is {:7.4f}".format(diff_score))
        if diff_score <= epsilon:
            break
        old_sense_vecs = deepcopy(new_sense_vecs)

    print("Finished running retrofitting.")

    return new_sense_vecs


if __name__ == "__main__":
    # parse command line input
    command_parse = read_args(sys.argv)
    # failed command line input
    if command_parse == 2:
        sys.exit(2)

    # try opening the specified files
    # try:
    vectors, v_dim = read_emb_in(command_parse[0])
    sense_voc, onto_adjacency = read_ontology(command_parse[1], vectors)
    n_iters = command_parse[3]
    epsilon = command_parse[4]
    # except:
    #     print("ERROR opening files. One of the paths or formats of the specified files was incorrect.")
    #     sys.exit(2)

    # run retrofitting and write to output file
    write_emb_out(retrofit(vectors, v_dim, sense_voc, onto_adjacency, n_iters, epsilon),
                  v_dim, command_parse[2])

    print("All done!")
