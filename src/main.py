import os
import pandas as pd
import time
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import lstm
import rnn
import utils
import lstm_train as lt

DATA_FILENAME = 'billboard_lyrics_1964-2015.csv'
YEAR_ERROR = 2
RNN_N_HIDDEN = 128
N_CATEGORIES = 51 # 51 years in the dataset


def main():
    # File I/O
    root_dir = os.path.abspath('..')
    data_dir = os.path.join(root_dir, 'data')
    os.path.exists(root_dir), os.path.exists(data_dir)

    # Option handling
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf", help="Run linear and ridge regression using TF-IDF",
                        action="store_true")
    parser.add_argument("--lstm", help="Run word-level LSTM",
                        action="store_true")
    args = parser.parse_args()

    ################################ "Naive" tf-idf analysis ####################################
    if (args.tfidf):
        # Load dataset
        full_dataset = pd.read_csv(os.path.join(data_dir, DATA_FILENAME), encoding='latin-1')
        x_train, x_test, y_train, y_test = train_test_split(list(full_dataset['Lyrics'].values.astype('U')),
                                                            list(full_dataset['Year']))
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

    ################################ LSTM analysis ####################################
    if (args.lstm):
        song_dict = utils.groupSongs(os.path.join(data_dir, DATA_FILENAME)) # 4634 rows after getting rid of NAs (batch = 2317)
        encoded_years, encoded_lyrics, vocab_len = lt.TokenizeDataset(song_dict)

        # Set up the training - use minibatch
        n_epochs = 10
        batch_size = 16 #100  # 11 batches of size 337 so iters = 11 (11 * 337 = 3707)

        # Split into training, validation, testing - train= 80% | valid = 10% | test = 10%
        split_frac = 0.8
        train_x = encoded_lyrics[0:int(split_frac * len(encoded_lyrics))] # 3707 training samples
        train_y = encoded_years[0:int(split_frac * len(encoded_lyrics))]  # 3707 training samples

        remaining_x = encoded_lyrics[int(split_frac * len(encoded_lyrics)):]
        remaining_y = encoded_years[int(split_frac * len(encoded_lyrics)):]

        valid_x = remaining_x[0:int(len(remaining_x) * 0.5)]
        valid_y = remaining_y[0:int(len(remaining_y) * 0.5)]

        test_x = remaining_x[int(len(remaining_x) * 0.5):]
        test_y = remaining_y[int(len(remaining_y) * 0.5):]

        # Dataloaders and batching
        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

        output_size = 51
        embedding_dim = 400
        hidden_dim = 128 #256
        n_layers = 2
        lstmc = lstm.LyricLSTM(vocab_len, output_size, embedding_dim, hidden_dim, n_layers)

        # Loss function + accuracy reporting
        current_loss = 0
        losses = np.zeros(n_epochs)  # For plotting
        accuracy = np.zeros(n_epochs)

        lr = 0.01
        criterion = nn.CrossEntropyLoss() #nn.BCELoss()
        optimizer = torch.optim.Adam(lstmc.parameters(), lr=lr)
        counter = 0
        print_every = 1
        clip = 5  # gradient clipping

        # Main training loop
        start = time.time()
        lstmc.train()
        for epoch in range(0, n_epochs):
            # initialize hidden state
            h = lstmc.init_hidden(batch_size)

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                #lstmc.zero_grad()
                optimizer.zero_grad()

                # get the output from the model
                #inputs = inputs.type(torch.LongTensor)
                output, h = lstmc(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.long())#.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(lstmc.parameters(), clip)
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = lstmc.init_hidden(batch_size)
                    val_losses = []
                    lstmc.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        inputs = inputs.type(torch.LongTensor)
                        output, val_h = lstmc(inputs, val_h) #TODO: problem here
                        val_loss = criterion(output.squeeze(), labels.long())#.float())

                        val_losses.append(val_loss.item())

                    lstmc.train()
                    print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

        # Get test data loss and accuracy

        test_losses = []  # track loss
        num_correct = 0

        # init hidden state
        h = lstmc.init_hidden(batch_size)

        lstmc.eval()
        # iterate over test data
        for inputs, labels in test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # get predicted outputs
            inputs = inputs.type(torch.LongTensor)
            output, h = lstmc(inputs, h)

            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer
            print(pred.size())
            print("Prediction: ")
            print(yearFromOutput(pred))

            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy())
            num_correct += np.sum(correct)

        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))

        # Plot losses and accuracy indexed over each epoch for training
        xaxis = np.arange(1, n_epochs+1)
        #utils.plotAccuracy(xaxis, accuracy)
        #utils.plotLoss(xaxis, losses)

if __name__ == "__main__":
    main()