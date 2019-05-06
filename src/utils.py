import torch, string
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

def yearFromOutput(output, songdict):
    """ Formats the output to consist of just the predicted year. """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return int(category_i) + 1965