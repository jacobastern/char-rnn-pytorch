from torch import nn
import torch

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """A gated recurrent unit, described here: https://en.wikipedia.org/wiki/Gated_recurrent_unit
        Args:
            input_size (int): the dimension of the input of the GRU.
            hidden_size (int): the dimension of the hidden state outputs of the GRU. 
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.w = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x_input, prev_hidden):
        """A forward pass of an input through the model. Follows the 'fully-gated' GRU architecture
        Args:
            x_input ((self.input_size) torch.Tensor): an integer-encoded token
            prev_hidden ((self.hidden_size) torch.Tensor): the hidden layer output of the previous GRU
        Returns:
            ht ((self.hidden_size) torch.Tensor): the resulting hidden layer representation
            ht ((self.hidden_size) torch.Tensor): another copy of the hidden layer, which will go
                through one more transformation to become the output token prediction.
        """
        zt = self.sig(self.wz(torch.cat((x_input, prev_hidden), dim=2)))
        rt = self.sig(self.wr(torch.cat((x_input, prev_hidden), dim=2)))
        h_tilde = self.tanh(self.w(torch.cat((torch.mul(rt, prev_hidden), x_input), dim=2)))
        ht = torch.mul((1 - zt), prev_hidden) + torch.mul(zt, h_tilde)
        return ht, ht

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """A full recurrent neural network, built around a GRU.
        Args:
            input_size (int): the dimension of the input to the network - in the 
                case of our CharRNN, it is len(alphabet). 
            hidden_size (int): the dimension of the hidden state outputs of the GRU. 
            output_size (int): the dimension of the output of the network. Note that 
                the output size is the same as the input size.
            n_layers (int): the number of layers in the RNN
        """
        super(RNN, self).__init__()
        # Input: An integer encoding of the character
        self.input_size = input_size
        # Output: A categorical distribution over characters
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # An embedding layer is different from a linear layer because it provides
        # lookup capability -- each character has its own trained embedding. A
        # linear layer is different, as all input characters share the same weights
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        # The first argument is self.hidden_size because we've embedded the input to be the same size as the hidden_state
        self.gru = GRU(self.hidden_size, self.hidden_size)
        self.to_output_size = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1) #dim=1

    def forward(self, input_char, hidden_state):
        """A forward pass through the RNN.
        Args:
            input_char (int): an integer-encoded character
            hidden_state ((self.hidden_size) torch.Tensor): the hidden state output of the previous layer
        """
        embed = self.embedding(input_char).view(1,1,-1)
        output, hidden = self.gru(embed, hidden_state)
        output = self.relu(self.to_output_size(output))
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)