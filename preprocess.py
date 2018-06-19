import nltk
from collections import Counter
import pandas as pd
import numpy as np
from loading_util import *


    
    
GO = '_GO'
EOS = 'EOS' #Also works as PAD
UNK = 'UNK'

extra_tokens = [GO,EOS,UNK]

'''
Indexing starts from 1 till 400000 without adding extra tokens. Function adds
extra tokens inside the 3 dictionaries
Returns: start_token index, end_token index , unk_token index
'''
def add_extra_to_dict(word_to_index,index_to_word,word_to_vec_map,embedding_dim= 50):
    #Adding decoder starter token
    word_to_index[GO] = len(word_to_index) + 1
    index_to_word[len(word_to_index)] = GO

    #Adding end token
    word_to_index[EOS] = len(word_to_index) + 1
    index_to_word[len(word_to_index)] = EOS

    #Adding unknown token
    word_to_index[UNK] = len(word_to_index) + 1
    index_to_word[len(word_to_index)] = UNK
    
    #Adding vector maps
    word_to_vec_map[GO] = np.random.uniform(-1.0,1.0, size = embedding_dim )
    word_to_vec_map[EOS] = np.random.uniform(-1.0,1.0, size = embedding_dim )
    word_to_vec_map[UNK] = np.random.uniform(-1.0,1.0, size = embedding_dim )
    
    return word_to_index[GO],word_to_index[EOS],word_to_index[UNK]


'''
Preprocesses the decoder input and output: Adds GO and EOS and UNK 
'''
def fit_decoder_text(data,word_to_index,max_allowed_seq_length):
    sentence_indices_input = np.array([[]])
    sentence_indices_output = np.array([[]])
    for txt in data:
        txt =  GO + ' ' + txt.lower() + ' '+ EOS
        print('Added tags..')
        print(txt)
        words = nltk.word_tokenize(txt)
        print("After tokenization")
        print(words)
        words = [word for word in words if word.isalnum()]
        print('Selected as words:')
        print(words)
        seq_length = len(words)
        if max_allowed_seq_length is not None and seq_length > max_allowed_seq_length:
            seq_length = max_allowed_seq_length
            words = words[:seq_length - 1] + [EOS]
        else:
            words = words + [EOS]

        decoder_phrase = []
        for w in words:
            if w in word_to_index:
                decoder_phrase += [word_to_index[w]]
            else:
                decoder_phrase += [word_to_index[UNK]]
                
        np.append(sentence_indices_input,decoder_phrase[:-1])
        np.append(sentence_indices_output,decoder_phrase[1:])
        

    return np.asarray(sentence_indices_input), np.asarray(sentence_indices_output)


wi , iw, wv = read_glove_vecs()
_,_,_ = add_extra_to_dict(wi,iw,wv)
X, Y = read_csv()

dinput,doutput= fit_decoder_text(data= Y[1:],word_to_index = wi,max_allowed_seq_length = 15)
