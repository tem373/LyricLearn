import os
import pandas as pd
import time
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import lstm
import rnn
import utils

DATA_FILENAME = 'billboard_lyrics_1964-2015.csv'
YEAR_ERROR = 2
RNN_N_HIDDEN = 128
N_CATEGORIES = 51 # 51 years in the dataset


def main():
    # File I/O
    root_dir = os.path.abspath('..')
    data_dir = os.path.join(root_dir, 'data')
    os.path.exists(root_dir), os.path.exists(data_dir)

    # Load dataset
    full_dataset = pd.read_csv(os.path.join(data_dir, DATA_FILENAME), encoding='latin-1')
    x_train, x_test, y_train, y_test = train_test_split(list(full_dataset['Lyrics'].values.astype('U')), list(full_dataset['Year']))

    ################################ "Naive" tf-idf analysis ####################################
    # TODO: try different regularization strengths
    # TODO: try multiple train test split configurations
    # TODO: try tfidf with different n-grams

    # Text Preprocessing
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3), max_features=20000)

    # Create vectors from train and test data - fit_transform() for training and transform() for testing
    # fit_transform() is basically fit() followed by transform()
    train_vec = vectorizer.fit_transform(x_train) # Unicode column values from dataset
    test_vec = vectorizer.transform(x_test)


    # Build vanilla linear regression model
    reg = LinearRegression(fit_intercept=True) # Intercept should be 1989 (halfway between 1965 & 2015)
    reg.fit(train_vec, y_train)

    # Validate on test data
    reg_preds = reg.predict(test_vec)
    rounded_reg = [int(round(x)) for x in reg_preds]

    print("\n\n LINEAR REGRESSION \n\n")
    for i in range(0, int(len(rounded_reg)/50)):
        print("Actual: " + str(y_test[i]) + " Predicted: " + str(rounded_reg[i]) + " Error: " + str(abs(y_test[i] - rounded_reg[i])))

    # Build Ridge model (linear regression plus regularization)
    clf = Ridge(alpha=1.0, random_state=1)
    clf.fit(train_vec, y_train)

    # Validate on test data
    ridge_preds = clf.predict(test_vec)
    rounded_predictions = [int(round(x)) for x in ridge_preds]

    print("\n\n RIDGE REGRESSION \n\n")
    for i in range(0, int(len(rounded_predictions)/50)):
        print("Actual: " + str(y_test[i]) + " Predicted: " + str(rounded_predictions[i]) + " Error: " + str(abs(y_test[i] - rounded_predictions[i])))

    # Evaluate using sklearn metrics - root mean squared error
    print("Linear Regression Mean Squared Error:" + str(mean_squared_error(y_test, reg_preds)))
    print("Ridge Mean Squared Error:" + str(mean_squared_error(y_test, ridge_preds)))

    ################################ RNN analysis ####################################
    song_dict = utils.groupSongs(os.path.join(data_dir, DATA_FILENAME)) # 4634 rows after getting rid of NAs (batch = 2317)
    split_idx = int(len(song_dict) * 0.8) # Split into training and testing
    train_dict = dict(list(song_dict.items())[:split_idx]) # 3707 training samples
    test_dict = dict(list(song_dict.items())[split_idx:])  # 927 test samples

    rnnc = rnn.RNN(utils.n_letters, RNN_N_HIDDEN, N_CATEGORIES) # Initialize RNN class

    # Set up the training - use minibatch
    n_iters = 1000
    n_epochs = 100
    batch_size = 337 # 11 batches of size 337 so iters = 11 (11 * 337 = 3707)

    # Loss function + accuracy reporting
    current_loss = 0
    losses = np.zeros(n_epochs)  # For plotting
    accuracy = np.zeros(n_epochs)

    # Main training loop
    start = time.time()
    for epoch in range(1, n_epochs + 1):

        for iter in range(n_iters): #TODO: make this 11 - have each batch be fed in once per iter
            year, lyric, year_tensor, lyric_tensor = utils.randomTrainingExample(train_dict)
            if (year == '\"Year\"'):
                continue
            output, loss = utils.trainRNN(year_tensor, lyric_tensor, rnnc)

            year_guess = utils.yearFromOutput(output)
            correct = '✓' if int(year_guess) == int(year) else '✗ (%s)' % year
            print('Epoch: %d %d %d%% (%s) %.4f %s / %s %s' % (epoch, iter, iter / n_iters * 100, utils.timeSince(start), loss, lyric[0:25], year_guess, correct))

            losses[epoch] += loss  # Create loss array
            if (correct == '✓'):
                accuracy[epoch] += 1 # Add 1 for each correct guess

        accuracy[epoch] /= 1000

    xaxis = np.arange(1, n_epochs+1)
    utils.plotAccuracy(xaxis, accuracy)
    utils.plotLoss(xaxis, losses)

    #TODO: testing loop
    for lyric, year in test_dict:
        pass

if __name__ == "__main__":
    main()