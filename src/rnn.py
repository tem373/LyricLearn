import os, csv, unicodedata, torch, string
import pandas as pd
import torch.nn as nn
import torch.utils.data
from io import open

data_filename = 'billboard_lyrics_1964-2015.csv'
file = '../data/billboard_lyrics_1964-2015.csv'

alphabet = string.ascii_letters
num_letters = len(alphabet)


class RNN(nn.Module):
    """ Module is the base class for all Pytorch NN modules"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def groupSongs(filename):
    lines = open(filename, encoding='latin-1').read().strip().split('\n')
    song_dict = {}
    for line in lines:
        line_list = line.split(',')
        if line_list[4] != 'Lyrics':
            song_dict[line_list[4]] = line_list[3]
    return song_dict


def getLetterIndex(letter):
    return alphabet.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, num_letters)
    tensor[0][getLetterIndex(letter)] = 1
    return tensor


def lyricsToTensor(lyrics):
    tensor = torch.zeros(len(lyrics), 1, num_letters)
    for li, letter in enumerate(lyrics):
        tensor[li][0][getLetterIndex(letter)] = 1
    return tensor


# s_dict = groupSongs(file)
# f_key = list(s_dict.keys())[1]
# print(f_key + " Year: " + s_dict[f_key])
#
# sample = "this is a sample lyric"
# #print(lyricsToTensor(sample))
# print(lyricsToTensor(f_key))






