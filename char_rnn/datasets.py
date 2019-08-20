import string
import zipfile
import os
import random
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, chunk_len=200, file_name="michael-jackson.txt"):
        """Creates a Pytorch Dataset from a text corpus.
        Args:
            chunk_len (int): the length of each training segment of text
            file_name (str): the name of the file to train on. A full list of files is found here: https://www.kaggle.com/paultimothymooney/poetry
        """
        root = 'data/'
            
        text_files = os.listdir(root)
        self.training_file = text_files[text_files.index(file_name)]
        with open(os.path.join(root, self.training_file), encoding='utf-8') as file:
            self.training_file = file.read()
        self.segment_extractor = self.FileSegmentExtractor(self.training_file, chunk_len)

    @staticmethod
    def extract_zip(zip_path):
        """Extraxts a file in .zip format to the root directory
        Args:
            zip_path (str): the path to the zip file
        """
        print('Unzipping {}'.format(zip_path))
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))
    
    def __len__(self):
        return self.len
        
    class FileSegmentExtractor():
        def __init__(self, training_file, chunk_len):
            """Extracts chunk_len segments from the data for training
            Args:
                training_file (str): the training file path
                chunk_len (int): the number of characters in each chunk
            """
            self.chunk_len = chunk_len
            self.training_file = training_file
            self.file_len = len(self.training_file)
            # A string including all printable characters
            self.all_characters = string.printable
            self.n_characters = len(self.all_characters)
            
        def random_chunk(self):
            """Extracts a random chunk from the file
            Returns:
                (str): a string of length (chunk_len)
            """
            start_index = random.randint(0, self.file_len - self.chunk_len)
            end_index = start_index + self.chunk_len + 1
            return self.training_file[start_index:end_index]

        def char_tensor(self, string):
            """Converts characters in a string to a numerical index representing that character.
            Args:
                string (str): the string to convert
            Returns:
                tensor (torch.Tensor): a tensor containing the indices of each letter in the string
            """
            tensor = torch.zeros(len(string)).long()
            for c in range(len(string)):
                try:
                    tensor[c] = self.all_characters.index(string[c])
                except ValueError:
                    tensor[c] = self.all_characters.index(' ')
            return tensor

        
        def random_training_set(self):
            """Obtains a random set of data to train on.
            Returns:
                inp (torch.Tensor): a chunk of characters from the file
                target (torch.Tensor): the same chunk of characters offset by one
            """
            chunk = self.random_chunk()
            inp = self.char_tensor(chunk[:-1])
            target = self.char_tensor(chunk[1:])
            return inp, target