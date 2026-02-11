import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np 

def normalize_vector(dataset):
    normal = (dataset-dataset.mean())/dataset.std()
    return normal

def define_dict_w_prop(input_path):
    smiles = []
    prop = []
    with open(input_path, "r") as file:
        for line in file:
            columns = line.split()
            smiles.append(columns[0])
            prop.append(float(columns[1])) # if prop is a scalar 
    prop = np.array(prop)
    prop = normalize_vector(prop)
    length = [len(smile_ent) for smile_ent in smiles]
    # recording all the characters appearing in the dataset smiles (char list)
    chars = sorted(list(set(''.join(smiles)))+[' '])
    nchars = len(chars) # number of unique chars 
    max_len = max(length) # max string length 
    #Defining char dictionary
    char_to_idx = {ch:i  for i, ch in enumerate(chars)}    
    idx_to_char = {i:ch  for i, ch in enumerate(chars)}   
    return smiles, char_to_idx, idx_to_char, nchars, max_len, prop

def pad_smile(string, max_str_len):
    '''
    this adds padding to the string to make all structures the same size 
    '''
    if len(string) <= max_str_len:
            return string + " " * (max_str_len - len(string))
    else:
         print(' check the entry for the smiles. The smile length is larger than the allowed length cap')


def smiles_to_hot(smiles, seq_len, unique_chars, char_to_idx):
    no_of_examples = len(smiles)
    smiles = [pad_smile(i, seq_len) for i in smiles if pad_smile(i, seq_len)]

    X = np.zeros((no_of_examples, seq_len, unique_chars), dtype=np.float32) 
    # nr of smiles in the dataset, length og the largest string in the dataset, nr of chars in the dictionary 

    for smile_idx, smile in enumerate(smiles):
        for char_idx, char in enumerate(smile):
            try:
                X[smile_idx, char_idx, char_to_idx[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Invalid SMILES:", smile)
                raise e
    return X

def hot_to_smiles(X, idx_to_char):
    # converts one matrix into smiles (one smile entry)
    string = []
    for i in X:
        # i is the each of the rows that stand for a char 
        try: 
            ind_char = np.where(i==1)[0][0]
            string.append(idx_to_char[ind_char])
        except Exception as e:
            # print('Invalid molecule -- unassigned character in the string...\nTry a new matrix')
            string = [] # discards the chars in the string so far as the generated matrix has a null char 
            break 
    if string:
        string = ''.join(string).replace(' ','')
    return string 

class CustomDataset(Dataset):
    '''
    the parent Dataset has no constructor (__init__), there's nothing to initialize in the parent. 
    That's why you don't need to write super().__init__(). 
    '''
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class DatasetSplitter:
    def __init__(self, data, prop, percentile, batch_size):
        self.data = data
        self.prop = prop
        self.batch_size = batch_size
        self.percentile = percentile

        self.trainloader, self.testloader = self.create_loaders()

    def create_loaders(self):
        dataset = CustomDataset(self.data, self.prop)

        train_size = int(self.percentile * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False) # shuffle changes the batches in every epoch 
        testloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

        return trainloader, testloader
