import os
import pandas as pd
import torch
import torch.utils.data
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILENAME = 'billboard_lyrics_1964-2015.csv'

class RNN(torch.nn.Module):
    """ Module is the base class for all Pytorch NN modules"""
    def __init__(self):
        pass


class LSTM(torch.nn.Module):
    """ Add docstring :D"""
    def __init__(self):
        pass

# File I/O
root_dir = os.path.abspath('..')
data_dir = os.path.join(root_dir, 'data')
os.path.exists(root_dir), os.path.exists(data_dir)

# Load dataset
full_dataset = pd.read_csv(os.path.join(data_dir, DATA_FILENAME), encoding='latin-1')
train_len = int(0.8 * len(full_dataset))
test_len = len(full_dataset) - train_len
train, test = torch.utils.data.random_split(full_dataset, [train_len, test_len])

# "Naive" analysis with tfidf
vectorizer = TfidfVectorizer()

