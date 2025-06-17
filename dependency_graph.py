# -*- coding: utf-8 -*-

import numpy as np
import spacy
import scipy.spatial.distance as distance
import pickle
import argparse


from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def _load_word_vec(path, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_id = []
    id_vec = []

    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        word_id.append(word)
        id_vec.append(vec)
    return word_id, id_vec

# fname = './glove.42B.300d.txt'
# word_id, id_vec = _load_word_vec(fname, embed_dim=300)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            # child_vec = id_vec[word_id.index(child)]
            # print(child_vec)
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for idx, line in enumerate(lines[1:]):
        try:
            [sent, aspect, polarity] = line.split('\t')
        except:
            print('解析样本出现错误, 已忽略:', line.split('\t'))
            continue
        polarity = polarity.strip()
        aspect = aspect.strip()
        sent = sent.strip()
        text_left, aspect, text_right = [s.lower().strip() for s in sent.partition(aspect)]
        # adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        adj_matrix = dependency_adj_matrix(text_left + " " + aspect + " " + text_right)
        # print(adj_matrix)
        idx2graph[idx] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close()
    # fout = open(filename + '.graph', 'wb')
    # for i in range(0, len(lines), 3):
    #     text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
    #     aspect = lines[i + 1].strip()
    #     # adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
    #     adj_matrix = dependency_adj_matrix(text_left + " " + aspect + " " + text_right)
    #     idx2graph[i] = adj_matrix
    # pickle.dump(idx2graph, fout)
    # fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, help='path to dataset')
    opt = parser.parse_args()
    # process(opt.dataset)

    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    # process('./datasets/semeval14/Restaurants_Train_Gold.xml.tsv')
    # process('./datasets/semeval14/Restaurants_Test_Gold.xml.tsv')
    # process('./datasets/semeval14/Laptops_Train.xml.seg')
    # process('./datasets/semeval14/Laptops_Test_Gold.xml.seg')
    # process('./datasets/semeval15/restaurant_train.tsv')
    # process('./datasets/semeval15/restaurant_test.tsv')
    process('./datasets/semeval16/restaurant_train.tsv')
    process('./datasets/semeval16/restaurant_test.tsv')

