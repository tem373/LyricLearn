import os
import pandas as pd
import torch
import torch.utils.data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import lstm
import rnn

DATA_FILENAME = 'billboard_lyrics_1964-2015.csv'
YEAR_ERROR = 2

# File I/O
root_dir = os.path.abspath('..')
data_dir = os.path.join(root_dir, 'data')
os.path.exists(root_dir), os.path.exists(data_dir)

# Load dataset
full_dataset = pd.read_csv(os.path.join(data_dir, DATA_FILENAME), encoding='latin-1')
x_train, x_test, y_train, y_test = train_test_split(list(full_dataset['Lyrics'].values.astype('U')), list(full_dataset['Year']))

################################ "Naive" tf-idf analysis ####################################

# Text Preprocessing
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3), max_features=20000)

# Create vectors from train and test data - fit_transform() for training and transform() for testing
# fit_transform() is basically fit() followed by transform()
train_vec = vectorizer.fit_transform(x_train) # Unicode column values from dataset
test_vec = vectorizer.transform(x_test)

# Build Ridge Model (linear regression plus regularization)
clf = Ridge(alpha=1.0, random_state=1)
clf.fit(train_vec, y_train)

# Validate on test data
predictions = clf.predict(test_vec)
rounded_predictions = [int(round(x)) for x in predictions]

for i in range(0, len(rounded_predictions)):
    print("Actual value: " + str(y_test[i]))
    print("Predicted value: " + str(rounded_predictions[i]))
    print("Error: " + str(abs(y_test[i] - rounded_predictions[i])))
    print("")

# Evaluate using sklearn metrics