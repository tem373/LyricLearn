import os, csv, unicodedata, torch
import pandas as pd
import torch.utils.data
from io import open

data_filename = 'billboard_lyrics_1964-2015.csv'
file = '../data/billboard_lyrics_1964-2015.csv'


class RNN(torch.nn.Module):
    """ Module is the base class for all Pytorch NN modules"""
    def __init__(self):
        pass


class LSTM(torch.nn.Module):
    """ Add docstring :D"""
    def __init__(self):
        pass

# Creates a dict data structure mapping each song to the year it was produced
def groupSongs(filename):
    lines = open(filename, encoding='latin-1').read().strip().split('\n')
    song_dict = {}
    for line in lines:
        line_list = line.split(',')
        if line_list[4] != 'Lyrics':
            song_dict[line_list[4]] = line_list[3]
    return song_dict

# How to convert the lyrics to tensors
def lyricsToTensor(lyrics):
    lyric_list = lyrics.split()
    lyric_tensor = torch.cuda.FloatTensor(lyric_list)
    return lyric_tensor

s_dict = groupSongs(file)
f_key = list(s_dict.keys())[1]
#print(f_key + " Year: " + s_dict[f_key])

sample = "this is a sample lyric"
print(lyricsToTensor(sample))




