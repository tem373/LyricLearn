import numpy as np

from collections import Counter

def TokenizeDataset(songdict):
    """ Takes a dict of the song lyrics to be tokenized and outputs the encoded + tokenized lyrics plus years/labels """
    lyrics_raw = []
    years_raw = []
    for lyric, year in songdict.items():
        lyrics_raw.append(lyric)
        years_raw.append(int(year)-1965)

    # Create vocab to int
    all_text = ' '.join(lyrics_raw)
    # create a list of words
    words = all_text.split()
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    # create mapping dictionary
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

    # Replace lyrics with ints in the songs
    lyrics_int = []
    for lyric in lyrics_raw:
        l = [vocab_to_int[w] for w in lyric.split()]
        lyrics_int.append(l)

    #TODO: try without encoding labels?
    encoded_years = np.array(years_raw)

    # NOTE: max song length is 1158 so make seq_length = 1150
    seq_length = 1000
    padded_lyrics = pad_features(lyrics_int, seq_length)

    return encoded_years, padded_lyrics, len(vocab_to_int)+1 # +1 for padding

def pad_features(lyrics_int, seq_length):
    ''' Return features of lyrics_ints, where each lyrics is padded with 0's or truncated to the input seq_length. '''
    features = np.zeros((len(lyrics_int), seq_length), dtype=int)

    for i, lyric in enumerate(lyrics_int):
        lyric_len = len(lyric)

        if lyric_len <= seq_length:
            zeroes = list(np.zeros(seq_length - lyric_len))
            new = zeroes + lyric

        elif lyric_len > seq_length:
            new = lyric[0:seq_length]

        features[i, :] = np.array(new)

    return features


