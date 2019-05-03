import os
import pandas as pd
import torch
import torch.utils.data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

DATA_FILENAME = 'billboard_lyrics_1964-2015.csv'

# File I/O
root_dir = os.path.abspath('..')
data_dir = os.path.join(root_dir, 'data')
os.path.exists(root_dir), os.path.exists(data_dir)

# Load dataset
full_dataset = pd.read_csv(os.path.join(data_dir, DATA_FILENAME), encoding='latin-1')
train_len = int(0.8 * len(full_dataset))
test_len = len(full_dataset) - train_len
train_rnn, test_rnn = torch.utils.data.random_split(full_dataset, [train_len, test_len])
#train_sk, test_sk = 

# "Naive" analysis with tfidf word frequency
# Do linear regression using the Tf-idf as a feature
vectorizer = TfidfVectorizer()
corpus = list(full_dataset['Lyrics'].values.astype('U')) # Unicode column values from dataset
X = vectorizer.fit_transform(corpus)

# Use ridge regression (and regular linear regression?) - ridge regression helps eliminate collinearity


