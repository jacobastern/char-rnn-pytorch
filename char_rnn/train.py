from torch import nn
from torch import optim
import torch
import matplotlib.pyplot as plt
import string
import time
# Local imports
from .model import RNN
from .datasets import TextDataset

def train(n_epochs=5000, print_every=500, plot_every=10, hidden_size=100, n_layers=1, lr=0.005, file_name="michael-jackson.txt"):
    """Trains a RNN according to the given parameters
    Args:
        n_epochs (int): the number of epochs to train for
        print_every (int): how often to print an evaluation string
        plot_every (int): how often to track the training loss
        hidden_size (int): the hidden dimension of the recurrent unit
        n_layers (int): the number of layers in the model
        lr (float): the learning rate for the model
        file_name (str): the name of the file to train on. Options: ['al-green.txt', 'Kanye_West.txt', 
            'britney-spears.txt', 'kanye-west.txt', 'notorious-big.txt', 'patti-smith.txt', 'prince.txt', 
            'leonard-cohen.txt', 'dolly-parton.txt', 'janisjoplin.txt', 'amy-winehouse.txt', 'dr-seuss.txt', 
            'rihanna.txt', 'adele.txt', 'eminem.txt', 'bjork.txt', 'radiohead.txt', 'missy-elliott.txt', 
            'beatles.txt', 'bruce-springsteen.txt', 'Lil_Wayne.txt', 'nickelback.txt', 'blink-182.txt', 
            'drake.txt', 'joni-mitchell.txt', 'bob-marley.txt', 'nicki-minaj.txt', 'lady-gaga.txt', 
            'kanye.txt', 'lorde.txt', 'bob-dylan.txt', 'lil-wayne.txt', 'dickinson.txt', 'bruno-mars.txt', 
            'alicia-keys.txt', 'r-kelly.txt', 'ludacris.txt', 'bieber.txt', 'nursery_rhymes.txt', 
            'michael-jackson.txt', 'dj-khaled.txt', 'lin-manuel-miranda.txt', 'paul-simon.txt', 'cake.txt', 
            'johnny-cash.txt', 'notorious_big.txt', 'nirvana.txt', 'jimi-hendrix.txt', 'disney.txt']
    """    
    all_characters = string.printable
    in_size=len(all_characters)
    output_size = len(all_characters)
    
    train_dataset = TextDataset(file_name=file_name)
    
    decoder = RNN(in_size, hidden_size, output_size, n_layers=n_layers)
    
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    all_losses = []
    running_loss = 0
    start = time.time()
    for epoch in range(n_epochs + 1):
        
        input_string, target_string = train_dataset.segment_extractor.random_training_set()
        loss_ = step(input_string, target_string, decoder, decoder_optimizer, criterion)
        running_loss += loss_
        
        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time.time() - start, epoch, epoch / n_epochs * 100, loss_))
            print(evaluate(decoder, 'Wh', 100), '\n')
            
        if epoch % plot_every == 0:
            all_losses.append(running_loss / (epoch + 1))
            
    plt.plot(range(len(all_losses)), all_losses, label='Loss')
    plt.xlabel("Epoch / {}".format(plot_every))
    plt.ylabel("Loss")

def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    """Samples from the trained RNN
    Args:
        decoder (torch.nn.Module): the RNN
        prime_str (str): a string to prime the model
        predict_len (int): the number of characters to predict
        temperature (float in [0, 1]): the amount of randomness in the sample
    Returns:
        predicted (str): a predicted string
    """
    hidden = decoder.init_hidden()
    predicted = prime_str
    prime_input = char_tensor(prime_str)
    all_characters = string.printable  

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):

        output, hidden = decoder(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        char_choice = all_characters[top_i]
        inp = char_tensor(char_choice)
        predicted += char_choice

    return predicted 

def step(input_string, target_string, decoder, decoder_optimizer, criterion):
    """Takes one step of training
    Args:
        input_string (torch.Tensor): a string of characters each encoded to a unique integer
            and stored in a tensor
        target_string (torch.Tensor): a string of characters each encoded to a unique integer
            and stored in a tensor. The same as the input string, but offset by one.
        decoder (torch.nn.Module): the RNN model
        decoder_optimizer (torch.optim.Optimizer): the optimizer for the model
        criterion (torch.nn.modules.loss._Loss): the objective function
    Returns:
        (torch.Tensor): a zero-dimensional tensor holding the loss-per-character
    """
    # initialize hidden layers, set up gradient and loss
    loss = 0
    hidden = decoder.init_hidden()
    num_classes = len(string.printable)
    i = 0
    decoder_optimizer.zero_grad()
    
    for in_char, target_char in zip(input_string, target_string):
        
        
        char_hat, hidden = decoder(in_char, hidden)
        target_char = target_char.unsqueeze(0)
        loss += criterion(char_hat.squeeze(0), target_char)
    
        i += 1
        
    loss.backward()
    decoder_optimizer.step()
        
    return loss.item() / len(input_string)
        
def char_tensor(chars):
    """Converts characters in a string to a numerical index representing that character.
    Args:
        chars (str): the string to convert
    Returns:
        tensor (torch.Tensor): a tensor containing the indices of each letter in the string
    """
    all_characters = string.printable    
    tensor = torch.zeros(len(chars)).long()
    for c in range(len(chars)):                
        tensor[c] = all_characters.index(chars[c])
    return tensor