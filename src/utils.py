import torch
import string
import random
import math
import time
from io import open

alphabet = string.ascii_letters
n_letters = len(alphabet)


def groupSongs(filename):
    lines = open(filename, encoding='latin-1').read().strip().split('\n')
    song_dict = {}
    for line in lines:
        line_list = line.split(',')
        # Throwaway label & get rid of unavailable lyrics
        if line_list[4] != 'Lyrics' and line_list[4] not in ["  ", "NA", "instrumental"]:
            song_dict[line_list[4]] = line_list[3]
    return song_dict


def getLetterIndex(letter):
    return alphabet.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][getLetterIndex(letter)] = 1
    return tensor


def lyricsToTensor(lyrics):
    tensor = torch.zeros(len(lyrics), 1, n_letters)
    for li, letter in enumerate(lyrics):
        tensor[li][0][getLetterIndex(letter)] = 1
    return tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def yearFromOutput(output, songdict):
    """ Formats the output to consist of just the predicted year. """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return int(category_i) + 1965


def randomTrainingExample(song_dict):
    lyric, year = random.choice(list(song_dict.items()))
    year_tensor = torch.tensor([int(year) - 1965], dtype=torch.long)
    lyric_tensor = lyricsToTensor(lyric)
    return year, lyric, year_tensor, lyric_tensor


def trainRNN(category_tensor, line_tensor, rnn):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = rnn.criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-rnn.learning_rate, p.grad.data)
    return output, loss.item()