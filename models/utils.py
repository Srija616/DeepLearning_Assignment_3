
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Vocab:
    def __init__(self, file_path, src_lang, tgt_lang):
        self.data = pd.read_csv(file_path, sep = ',', header = None, names = [src_lang, tgt_lang])
        self.data.dropna()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        src_chars = sorted(set(''.join(self.data[src_lang])))
        tgt_chars = sorted(set(''.join(self.data[tgt_lang])))

        self.src_char_to_idx = {src_chars[i]:i+3 for i in range(len(src_chars))}
        self.src_char_to_idx['<'] = 0
        self.src_char_to_idx['<pad>'] = 1
        self.src_char_to_idx['<unk>'] = 2

        self.src_vocab = {src_chars[i]:i+3 for i in range(len(src_chars))}
        self.src_vocab['<'] = 0
        self.src_vocab['<pad>'] = 1
        self.src_vocab['<unk>'] = 2
        

        
        self.src_idx_to_char = {idx: char for char, idx in self.src_char_to_idx.items()}
 
        self.tgt_char_to_idx =  {tgt_chars[i]:i+3 for i in range(len(tgt_chars))}
        self.tgt_char_to_idx['<'] = 0
        self.tgt_char_to_idx['<pad>'] = 1
        self.tgt_char_to_idx['<unk>'] = 2

        self.tgt_vocab =  {tgt_chars[i]:i+3 for i in range(len(tgt_chars))}
        self.tgt_vocab['<'] = 0
        self.tgt_vocab['<pad>'] = 1
        self.tgt_vocab['<unk>'] = 2


        self.tgt_idx_to_char = {idx: char for char, idx in self.tgt_char_to_idx.items()}
        # self.src_vocab = (self.src_char_to_idx)
        # self.tgt_vocab = (self.tgt_char_to_idx)
        print (len(self.src_vocab), len(self.tgt_vocab))

    def get(self):
        return self.src_vocab, self.tgt_vocab, self.src_char_to_idx, self.src_idx_to_char, self.tgt_char_to_idx, self.tgt_idx_to_char 


class TransliterationDataLoader(Dataset):
    """
    A PyTorch Dataset class for loading transliteration data.

    Args:
        filename (str): Path to the CSV file containing transliteration pairs.
        src_lang (str): Name of the source language column in the CSV file.
        tgt_lang (str): Name of the target language column in the CSV file.
        src_vocab (dict): Mapping of characters in the source language to their indices.
        tgt_vocab (dict): Mapping of characters in the target language to their indices.
        src_char_to_idx (dict): Mapping of characters in the source language to their indices.
        tgt_char_to_idx (dict): Mapping of characters in the target language to their indices.
    """

    def __init__(self, filename, src_lang, tgt_lang, src_vocab, tgt_vocab, src_char_to_idx, tgt_char_to_idx):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_char_to_idx = src_char_to_idx
        self.tgt_char_to_idx = tgt_char_to_idx
        self.start_of_word = 0
        self.data = pd.read_csv(filename, sep = ',', header = None, names = [self.src_lang, self.tgt_lang])
        
        self.src_dim = max([len(word) for word in self.data[self.src_lang].values]) + 1
        self.tgt_dim = max([len(word) for word in self.data[self.tgt_lang].values]) + 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset based on the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the English word and Hindi word.
        """

        # print (self.data.head(5))
        # print (self.data.columns)
        # print (idx)
        src_word = self.data.iloc[idx][self.src_lang]
        tgt_word = self.data.iloc[idx][self.tgt_lang]

        # print (src_word, tgt_word)
        # src_word = self.data.loc[self.src_lang, idx]
        # tgt_word = self.data.loc[self.tgt_lang, idx]

        # get the index of characters for the given src word and tgt word. If no mapping exists, give the value for the <unk> - unknown token
        src_indices = [self.src_vocab.get(char, self.src_vocab['<unk>']) for char in src_word]
        tgt_indices = [self.tgt_vocab.get(char, self.tgt_vocab['<unk>']) for char in tgt_word]

        src_indices.insert(0, self.start_of_word)
        tgt_indices.insert(0, self.start_of_word)
        
        src_len = len(src_indices)
        tgt_len = len(tgt_indices)
        
        src_pad = [self.src_vocab['<pad>']] * (self.src_dim - src_len)
        tgt_pad = [self.tgt_vocab['<pad>']] * (self.tgt_dim - tgt_len)

        src_indices.extend(src_pad)
        tgt_indices.extend(tgt_pad)

        src_tensor = torch.LongTensor(src_indices)
        tgt_tensor = torch.LongTensor(tgt_indices)

        return src_tensor, tgt_tensor, src_len, tgt_len



def get_word_from_index(tgt, tgt_idx_to_char, gt=True):
    batch_size = tgt.shape[0]
    char_seq = tgt.shape[1]
    strings = []
    for i in range (batch_size):
        chars = []
        for j in range (char_seq):
            if gt:
                if tgt[i, j].item() not in [0, 1]:
                    chars.append(tgt_idx_to_char[tgt[i,j].item()])
            # else:
            #     # if the tgt is predicted, not ground truth then keep the pad tokens
            #     if tgt[i, j].item() not in [0]:
            #         chars.append(tgt_idx_to_char[tgt[i,j].item()])
        string = ''.join(chars)
        strings.append(string)
    
    return strings

def calculate_accuracy(model, data_loader, tgt_idx_to_char, loss_criterion, log_to_file = False):
    
    correct_predictions = 0
    total = 0
    
    epoch_loss = 0
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(data_loader):
            # Convert target indices to string for comparison
            tgt_string=get_word_from_index(tgt,tgt_idx_to_char, True)
            # print (tgt_string)
            # print (len(tgt_string))
            # print ('String tgt', tgt_string)
            # print (f'Point 1: {tgt.shape}')
            # Move tensors to the device
            src = src.permute(1, 0)
            tgt = tgt.permute(1, 0)
            # print (f'Point 2: {tgt.shape}')
            src = src.to(device)
            tgt = tgt.to(device)

            output = model(src, tgt, 0)
            
            output = output[1:].reshape(-1, output.shape[2])

            tgt = tgt[1:].reshape(-1)

            # exit()
            # Calculate the loss
            output = output.to(device)
            # print (output.shape)

            mask = (output!=1)
            output_masked = output*mask

            # print ('mask: ', mask.shape)

            loss = loss_criterion(output_masked, tgt)
            epoch_loss += loss.item()
            
            batch_size = tgt_len.shape[0]
            seq_length = int(tgt.numel() / batch_size)


            # Convert the output to predicted characters
            predicted_indices = torch.argmax(output, dim=1)
            predicted_indices = predicted_indices.reshape(seq_length,-1)
            predicted_indices = predicted_indices.permute(1, 0)
            # Convert predicted indices to strings
            string_pred=get_word_from_index(predicted_indices,tgt_idx_to_char, True)
            for i in range(batch_size):
                if (log_to_file):
                    predictions.append([string_pred[i], tgt_string[i]])
                total+=1
                # Compare the predicted string with the target string
                # predicted_string = string_pred[i][:len(tgt_string[i])]
                # print (predicted_string)
                # print (tgt_string[i])
                # print (len(predicted_string), len(tgt_string))
                # exit()
                # print (predicted_string, tgt_string[i])

                if string_pred[i] == tgt_string[i]:
                    correct_predictions+=1
    # exit()
    print("Total",total)
    print("Correct",correct_predictions)
    loss_out = (epoch_loss/batch_size)
    accuracy = (correct_predictions /total) * 100            

    return loss_out, accuracy, predictions


def dataloader(train_csv, val_csv, test_csv, batch_size):
    '''
    Load data into batches.
    Arguments: 
    training data csv path
    validation data csv path
    test data csv path
    batch_size

    Returns:
    train, val, test dataloaders, idx to character mappings
    '''
    src_lang = 'English'
    tgt_lang = 'Hindi'
    vocab = Vocab(train_csv, src_lang, tgt_lang)
    src_vocab,tgt_vocab, src_char_to_idx, src_idx_to_char, tgt_char_to_idx, tgt_idx_to_char  = vocab.get()

    train_dataset = TransliterationDataLoader(train_csv, src_lang, tgt_lang, src_vocab, tgt_vocab, src_char_to_idx, tgt_char_to_idx)
    val_dataset = TransliterationDataLoader(val_csv, src_lang, tgt_lang, src_vocab, tgt_vocab, src_char_to_idx, tgt_char_to_idx)
    test_dataset = TransliterationDataLoader(test_csv, src_lang, tgt_lang, src_vocab, tgt_vocab, src_char_to_idx, tgt_char_to_idx)
    
    print ('Dataset created successfully!')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    
    print ('Data loaded successfully!')

    return train_loader,val_loader, test_loader, src_idx_to_char, tgt_idx_to_char