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
import argparse
import gzip
import os
from copy import deepcopy

import numpy as np
from scipy.sparse import lil_matrix


SENSE_SEP = '%'
VAL_SEP = '#'


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


def retrofit(wvecs, v_dim, sense_voc, onto_adjacency, n_iters, threshold):
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
    wvecs, v_dim = read_emb_in(argp.in_fpath)
    sense_voc, onto_adjacency = read_ontology(argp.onto_fpath, wvecs)

    if argp.out_fpath is None:
        out_fpath = os.path.join('output',
                                 os.path.splitext(os.path.basename(argp.in_fpath))[0])
    else:
        out_fpath = argp.out_fpath

    # run retrofitting and write to output file
    new_sense_vecs = retrofit(wvecs, v_dim, sense_voc, onto_adjacency,
                              argp.n_iters, argp.threshold)
    write_emb_out(new_sense_vecs, v_dim, out_fpath)

    print("All done!")


if __name__ == "__main__":
    main(parse_args())
