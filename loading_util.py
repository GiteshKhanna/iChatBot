import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from preprocess import add_extra_to_dict

def read_glove_vecs(glove_file="./glove.6B.50d.txt"):
    with open(glove_file, 'r', encoding = 'utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 0
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def read_csv(filename = 'data/data_10.csv'):
    ans = []
    ques = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            ans.append(row[0])
            ques.append(row[1])
    #print(type(ans))
    X = np.asarray(ans)
    #print(type(a))
    Y = np.asarray(ques)

    return X, Y

def map_dict_to_list(iw, wv):
    emb = []
    for idx in range(0,len(iw)):
        emb.append(wv[iw[idx]])
    return emb

def read_csv_essay(filename = 'data/essays.csv'):
    essays=[]
    traits=[]

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            essays.append(row[1])
            example = row[2:]
            example = [0 if(x=='n') else 1 for x in example]
            traits.append(example)

    essays = np.asarray(essays)
    traits = np.asarray(traits)

    return essays[1:],traits[1:]
    
            

'''
#Testing all
#Remember importing from other file
#read_csv()
wi,iw,wv = read_glove_vecs("./glove.6B.50d.txt")
add_extra_to_dict(wi,iw,wv)
emb = map_dict_to_list(iw,wv)
print(emb[0])
'''
