import os
import cv2
import numpy as np
import string
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import seaborn as sns
from IPython.display import clear_output

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.activations import elu
from tqdm import tqdm
from collections import Counter

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import random
from PIL import Image
import xml.etree.ElementTree as ET


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst


class My_Generator(Sequence):
    
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        batch_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_texts = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        training_txt = []
        train_label_length = []
        train_input_length = []

        for im_path, text in zip(batch_paths, batch_texts):
            
            try:
                text = str(text).strip()
                img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)   

                ### actually returns h, w
                h, w = img.shape

                ### if height less than 32
                if h < 32:
                    add_zeros = np.ones((32-h, w)) * 255
                    img = np.concatenate((img, add_zeros))
                    h = 32

                ## if width less than 128
                if w < 128:
                    add_zeros = np.ones((h, 128-w)) * 255
                    img = np.concatenate((img, add_zeros), axis=1)
                    w = 128

                ### if width is greater than 128 or height greater than 32
                if w > 128 or h > 32:
                    img = cv2.resize(img, (128, 32))

                img = np.expand_dims(img , axis = 2)

                # Normalize each image
                img = img / 255.

                images.append(img)
                training_txt.append(encode_to_labels(text))
                train_label_length.append(len(text))
                train_input_length.append(31)
            except:
                
                pass

        return [np.array(images), 
               pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list)), 
               np.array(train_input_length), 
               np.array(train_label_length)], np.zeros(len(images))


def image_text_model():
  vocab=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  char_list=sorted(vocab)
  inputs = Input(shape=(32, 128, 1))

  conv_1 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(inputs)
  pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
  
  conv_2 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(pool_1)
  pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

  conv_3 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(pool_2)
  conv_4 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(conv_3)
  pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
  
  conv_5 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(pool_4)
  batch_norm_5 = BatchNormalization()(conv_5)
  
  conv_6 = Conv2D(64, (3,3), activation = 'elu', padding='same',kernel_initializer='he_normal')(batch_norm_5)
  batch_norm_6 = BatchNormalization()(conv_6)
  pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
  
  conv_7 = Conv2D(64, (2,2), activation = 'elu',kernel_initializer='he_normal')(pool_6)
  
  squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

  bgru_1 = Bidirectional(CuDNNGRU(256, return_sequences=True))(squeezed)
  bgru_2 = Bidirectional(CuDNNGRU(256, return_sequences=True))(bgru_1)

  outputs = Dense(len(char_list) + 1, activation = 'softmax')(bgru_2)
  act_model = Model(inputs, outputs)
  return act_model,char_list,outputs,inputs


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def pre_process_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    if h < 32:
        add_zeros = np.ones((32-h, w)) * 255
        img = np.concatenate((img, add_zeros))
        h = 32
    if w < 128:
        add_zeros = np.ones((h, 128-w)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
        w = 128
    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img , axis = 2)
    img = img / 255.
    return img



# def predict_output(img):
#     prediction = act_model.predict(np.array([img]))
#     out = K.get_value(K.ctc_decode(prediction, 
#                                    input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
#                                    greedy=True)[0][0])
#     for x in out:
#         print("predicted text = ", end = '')
#         for p in x:
#             if int(p) != -1:
#                 print(char_list[int(p)], end = '')
#         print('\n')

def predict_output(img,act_model,char_list):
    pred_text = ""
    # act_model,char_list,_,_=image_text_model()
    prediction = act_model.predict(np.array([img]))
    out = K.get_value(K.ctc_decode(prediction,input_length=np.ones(prediction.shape[0]) * prediction.shape[1],greedy=True)[0][0])
    for x in out:
        for p in x:
            if int(p) != -1:
                pred_text=pred_text+char_list[int(p)]
    return pred_text



#####transliteration
# Remove all English non-letters
MAX_OUTPUT_CHARS = 30
non_eng_letters_regex = re.compile('[^a-zA-Z ]')
pad_char = '-PAD-'
# kannada Unicode dec Range

kannada_alphabets = [chr(alpha) for alpha in range(3202, 3311)]
kannada_alphabet_size = len(kannada_alphabets)

kannada_alpha2index = {pad_char: 0}
for index, alpha in enumerate(kannada_alphabets):
    kannada_alpha2index[alpha] = index+1

# print(kannada_alpha2index)

def cleanEnglishVocab(line):
    line = line.replace('-', ' ').replace(',', ' ').upper()
    line = non_eng_letters_regex.sub('', line)
    return line.split()

# Remove all kannada non-letters
def cleankannadaVocab(line):
    line = line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in line:
        if char in kannada_alpha2index or char == ' ':
            cleaned_line += char
    return cleaned_line.split()


def word_rep(word, letter2index, device = 'cpu'):
    rep = torch.zeros(len(word)+1, 1, len(letter2index)).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index][0][pos] = 1
    pad_pos = letter2index[pad_char]
    rep[letter_index+1][0][pad_pos] = 1
    return rep

def gt_rep(word, letter2index, device = 'cpu'):
    gt_rep = torch.zeros([len(word)+1, 1], dtype=torch.long).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        gt_rep[letter_index][0] = pos
    gt_rep[letter_index+1][0] = letter2index[pad_char]
    return gt_rep


class Transliteration_EncoderDecoder_Attention(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, verbose=False):
        super(Transliteration_EncoderDecoder_Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_rnn_cell = nn.GRU(input_size, hidden_size)
        self.encoder_rnn_cell2 = nn.GRU(hidden_size, hidden_size)
        self.decoder_rnn_cell = nn.GRU(hidden_size*2, hidden_size)
        self.decoder_rnn_cell2 = nn.GRU(hidden_size*2, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        self.U = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, 1)
        self.out2hidden = nn.Linear(self.output_size, self.hidden_size)   
        
        self.verbose = verbose
        
    def forward(self, input, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        
        # encoder
        encoder_outputs, hidden = self.encoder_rnn_cell(input)
        
        # encoder_outputs = encoder_outputs.view(-1, self.hidden_size)
        encoder_outputs, hidden = self.encoder_rnn_cell2(encoder_outputs)
        encoder_outputs = encoder_outputs.view(-1, self.hidden_size)
        
        if self.verbose:
            print('Encoder output', encoder_outputs.shape)
        
        # decoder
        decoder_state = hidden
        decoder_input = torch.zeros(1, 1, self.output_size).to(device)
        
        outputs = []
        U = self.U(encoder_outputs)
        
        if self.verbose:
            print('Decoder state', decoder_state.shape)
            print('Decoder intermediate input', decoder_input.shape)
            print('U * Encoder output', U.shape)
        
        for i in range(max_output_chars):
            
            W = self.W(decoder_state.view(1, -1).repeat(encoder_outputs.shape[0], 1))
            V = self.attn(torch.tanh(U + W))
            attn_weights = F.softmax(V.view(1, -1), dim = 1) 
            
            if self.verbose:
                print('W * Decoder state', W.shape)
                print('V', V.shape)
                print('Attn', attn_weights.shape)
            
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
            
            embedding = self.out2hidden(decoder_input)
            decoder_input = torch.cat((embedding[0], attn_applied[0]), 1).unsqueeze(0)
            
            if self.verbose:
                print('Attn LC', attn_applied.shape)
                print('Decoder input', decoder_input.shape)
                
            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)
            
            if self.verbose:
                print('Decoder intermediate output', out.shape)
                
            # out = self.h2o(decoder_state)
            # out, decoder_state = self.decoder_rnn_cell2(out, decoder_state)
            out = self.h2o(decoder_state)
            out = self.softmax(out)
            outputs.append(out.view(1, -1))
            
            if self.verbose:
                print('Decoder output', out.shape)
                self.verbose = False
            
            max_idx = torch.argmax(out, 2, keepdim=True)
            if not ground_truth is None:
                max_idx = ground_truth[i].reshape(1, 1, 1)
            one_hot = torch.zeros(out.shape, device=device)
            one_hot.scatter_(2, max_idx, 1) 
            
            decoder_input = one_hot.detach()
            
        return outputs

def infer(net, word, char_limit,eng_alpha2index, device = 'cpu'):
    input = word_rep(word, eng_alpha2index, device)
    return net(input, char_limit)

# eng_alpha2index = {pad_char: 0}
# for index, alpha in enumerate(eng_alphabets):
#     eng_alpha2index[alpha] = index+1

res = dict((v,k) for k,v in kannada_alpha2index.items())

def language_translation(net, word, eng_alpha2index,device = 'cpu'):
    net = net.eval().to(device)
    outputs = infer(net, word, 30, eng_alpha2index,device)
    kannada_output = ''
    for out in outputs:
        val, indices = out.topk(1)
        index = indices.tolist()[0][0]
        if index == 0:
            break
        # print(index)
        kannada_char = res[index]
        kannada_output += kannada_char
    # print(word + ' - ' + kannada_output)
    return kannada_output