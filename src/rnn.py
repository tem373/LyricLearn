import torch.nn as nn
import torch.utils.data
import torch.optim as optim

class RNN(nn.Module):
    """ Module is the base class for all Pytorch NN modules"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss() # could try criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.01 # If you set this too high, it might explode. If too low, it might not learn
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)