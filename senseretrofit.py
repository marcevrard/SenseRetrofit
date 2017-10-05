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
    ./senseretrofit.py -i sample_data/sample_vec.npy -q sample_data/sample_onto.txt.gz
'''
import argparse
import gzip
import os
from copy import deepcopy

import numpy as np
from scipy import sparse

import embedding_tools as emb

SENSE_SEP = '%'
VAL_SEP = '#'


def onto2word(onto_token):
    return onto_token.split(SENSE_SEP)[0]


def onto2sense(onto_token):
    return onto_token.split(VAL_SEP)[0]


def wvecs2embeds(wvecs):
    id2token, embeds = [], []
    for token in sorted(wvecs.keys()):
        id2token.append(token)
        embeds.append(wvecs[token])
    return id2token, np.array(embeds, dtype=np.float32)  # pylint: disable=no-member


def add_token2voc(token, onto_sense2id, onto_idx, emb_voc):
    ''' Add word sense tokens to a vocabulary relevant to the input vectors.'''
    # check if this sense has a corresponding word vector
    if onto2word(token) not in emb_voc:
        return onto_idx

    # check if the sense isn't already in the vocabulary
    if token not in onto_sense2id:
        onto_sense2id[token] = onto_idx
        return onto_idx + 1

    return onto_idx


def read_ontology(fpath, id2word):
    ''' Read the subset of the ontology relevant to the input vectors.'''

    print("Reading ontology from file...")
    if fpath.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    emb_voc = set(id2word)
    # Index all the word senses
    onto_sense2id = {}
    onto_idx = 0

    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for token in line:
                sense_token = onto2sense(token)
                onto_idx = add_token2voc(sense_token, onto_sense2id, onto_idx, emb_voc)
    onto_idx += 1

    # Create the sparse adjacency matrix of weights between senses
    adjacency_mtx = sparse.lil_matrix((onto_idx, onto_idx))
    with f_open(fpath) as f:
        for line in f:
            line = line.decode().strip().split()
            for idx, token in enumerate(line):
                sense_token, val = token.split(VAL_SEP)
                if sense_token in onto_sense2id:
                    # find the row index
                    if idx == 0:
                        row = onto_sense2id[sense_token]
                    # find the col index of the neighbor and set its weight
                    col = onto_sense2id[sense_token]
                    adjacency_mtx[row, col] = float(val)
                else:
                    if idx == 0:
                        break
                    continue

    print("Finished reading ontology.")

    id2onto_sense = [sense_token for sense_token, _idx
                     in sorted(onto_sense2id.items(), key=lambda x: x[1])]

    return id2onto_sense, adjacency_mtx.tocoo()


def max_vec_diff(new_vecs, old_vecs):
    ''' Return the maximum differential between old and new vectors
    to check for convergence.'''
    max_diff = 0.0
    for k in new_vecs:
        diff = np.linalg.norm(new_vecs[k] - old_vecs[k])
        if diff > max_diff:
            max_diff = diff
    return max_diff


def retrofit(word2id, embeds, id2onto_sense, onto_adjacency, n_iters, threshold):
    ''' Run the retrofitting procedure.'''
    print("Starting the retrofitting procedure...")

    # Get the word types in the ontology
    onto_words = set([onto2word(token) for token in id2onto_sense])
    # Initialize sense vectors to sense agnostic counterparts
    new_sense_vecs = {token: embeds[word2id[onto2word(token)]]
                      for token in id2onto_sense}
    # Create dummy sense vectors for words that aren't in the ontology (these won't be updated)
    new_sense_vecs.update({'{}{}0:00:00::'.format(word, SENSE_SEP): embeds[word2id[word]]
                           for word in word2id if word not in onto_words})

    # Create a copy of the sense vectors to check for convergence
    old_sense_vecs = deepcopy(new_sense_vecs)

    # Run for a maximum number of iterations
    for itr in range(n_iters):
        new_vec = None
        normalizer = None
        prev_row = None
        print("Running retrofitting iter {:2d}...".format(itr + 1), end=' ')
        # Loop through all the non-zero weights in the adjacency matrix
        for row, col, val in zip(onto_adjacency.row, onto_adjacency.col, onto_adjacency.data):
            if row != prev_row:  # New sense
                if prev_row:
                    new_sense_vecs[id2onto_sense[prev_row]] = new_vec / normalizer

                # Reinitialize vars
                new_vec = np.zeros(embeds.shape[1], dtype=float)
                normalizer = 0.0
                prev_row = row

            if row == col:  # Add the sense agnostic vector
                word = onto2word(id2onto_sense[row])
                new_vec += val * embeds[word2id[word]]
            else:  # Add a neighboring vector
                new_vec += val * new_sense_vecs[id2onto_sense[col]]

            normalizer += val

        diff_score = max_vec_diff(new_sense_vecs, old_sense_vecs)
        print("Max vector differential is {:7.4f}".format(diff_score))
        if diff_score <= threshold:
            break
        old_sense_vecs = deepcopy(new_sense_vecs)

    print("Finished running retrofitting.")

    return new_sense_vecs


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-fpath', required=True,
                        help="Embedding input file path (gz, txt or npy file formats).")
    parser.add_argument('-o', '--out-fpath',
                        help="Embedding output file path (optional, default: <in_fpath>_sense).")
    parser.add_argument('-q', '--onto-fpath',
                        help="Ontology input file path (gz or txt file formats).")
    parser.add_argument('-n', '--n_iters', type=int, default=10,
                        help="Number of iterations.")
    parser.add_argument('-t', '--threshold', type=float, default=0.001,
                        help="Convergence threshold.")
    parser.add_argument('-f', '--format', choices=['txt', 'npy'], default='npy',
                        help="Choice of output format.")

    return parser.parse_args()


def main(argp):
    id2word, embeds = emb.load_embeds_np(argp.in_fpath)
    id2onto_sense, onto_adjacency = read_ontology(argp.onto_fpath, id2word)

    word2id = {word: idx for idx, word in enumerate(id2word)}


    if argp.out_fpath is None:
        out_fpath = os.path.join('output',
                                 os.path.splitext(os.path.basename(argp.in_fpath))[0] + '_sense')
    else:
        out_fpath = argp.out_fpath

    # Run retrofitting and write to output file
    new_sense_vecs = retrofit(word2id, embeds, id2onto_sense, onto_adjacency,
                              argp.n_iters, argp.threshold)

    emb.save_embeds_np(*wvecs2embeds(new_sense_vecs), out_fpath)

    print("Done and saved in:", out_fpath)


if __name__ == "__main__":
    main(parse_args())
