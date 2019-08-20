import argparse
# Local imports
from char_rnn.train import train

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-e', '--n_epochs', required=False, type=int, default=5000, 
        help='The number of epochs to train for.')
    parser.add_argument('-f', '--file_name', required=False, type=str, default='michael-jackson.txt', 
        help='The name of the file to train on.',
        choices=['al-green.txt', 'Kanye_West.txt', 
            'britney-spears.txt', 'kanye-west.txt', 'notorious-big.txt', 'patti-smith.txt', 'prince.txt', 
            'leonard-cohen.txt', 'dolly-parton.txt', 'janisjoplin.txt', 'amy-winehouse.txt', 'dr-seuss.txt', 
            'rihanna.txt', 'adele.txt', 'eminem.txt', 'bjork.txt', 'radiohead.txt', 'missy-elliott.txt', 
            'beatles.txt', 'bruce-springsteen.txt', 'Lil_Wayne.txt', 'nickelback.txt', 'blink-182.txt', 
            'drake.txt', 'joni-mitchell.txt', 'bob-marley.txt', 'nicki-minaj.txt', 'lady-gaga.txt', 
            'kanye.txt', 'lorde.txt', 'bob-dylan.txt', 'lil-wayne.txt', 'dickinson.txt', 'bruno-mars.txt', 
            'alicia-keys.txt', 'r-kelly.txt', 'ludacris.txt', 'bieber.txt', 'nursery_rhymes.txt', 
            'michael-jackson.txt', 'dj-khaled.txt', 'lin-manuel-miranda.txt', 'paul-simon.txt', 'cake.txt', 
            'johnny-cash.txt', 'notorious_big.txt', 'nirvana.txt', 'jimi-hendrix.txt', 'disney.txt'])
    
    parser.add_argument('--print_every', required=False, type=int, default=500, 
                        help='How often to print an evaluation string.')
    parser.add_argument('--plot_every', required=False, type=int, default=10, 
                        help='How often to track the training loss.')
    parser.add_argument('--hidden_size', required=False, type=int, default=100, 
                        help='The hidden dimension of the recurrent unit.')
    parser.add_argument('--n_layers', required=False, type=int, default=1, 
                        help='The number of layers in the model.')
    parser.add_argument('--lr', required=False, type=int, default=.005, 
                        help='The learning rate for the model.')
    
    args = parser.parse_args()

    train(args.n_epochs, 
          args.print_every, 
          args.plot_every, 
          args.hidden_size,
          args.n_layers,
          args.lr,
          args.file_name)
    