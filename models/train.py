import pandas as pd
import os
import argparse
import wandb
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, random_split
) 
import torch.optim as optim
from utils import TransliterationDataLoader, Vocab
from model import Encoder, Decoder, EncoderDecoder
from tqdm import tqdm


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
            else:
                # if the tgt is predicted, not ground truth then keep the pad tokens
                if tgt[i, j].item() not in [0]:
                    chars.append(tgt_idx_to_char[tgt[i,j].item()])
        string = ''.join(chars)
        strings.append(string)
    
    return strings

def calculate_accuracy(model, data_loader, tgt_idx_to_char, loss_criterion):
    
    correct_predictions = 0
    total = 0
    
    epoch_loss = 0
    model.eval()
    
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
            string_pred=get_word_from_index(predicted_indices,tgt_idx_to_char, False)
            for i in range(batch_size):
                total+=1
                # Compare the predicted string with the target string
                predicted_string = string_pred[i][:len(tgt_string[i])]
                # print (predicted_string)
                # print (tgt_string[i])
                # print (len(predicted_string), len(tgt_string))
                # exit()
                # print (predicted_string, tgt_string[i])
                if predicted_string == tgt_string[i]:
                    correct_predictions+=1
    # exit()
    print("Total",total)
    print("Correct",correct_predictions)
    return (epoch_loss/(len(data_loader))), (correct_predictions /total) * 100


print ('Execution started')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print (device)

folder = './data/hin/'
train_csv = f'{folder}/hin_train.csv'
test_csv = f'{folder}/hin_test.csv'
valid_csv = f'{folder}/hin_valid.csv'

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

def train(args):
    print ('Started dataloading...')
    
    train_dataloader, val_dataloader, test_dataloader, src_idx_to_char, tgt_idx_to_char = dataloader(train_csv, valid_csv, test_csv, args.batch_size)
    
    INPUT_DIM = len(src_idx_to_char)  # vocab size for English : 26 + 1 (pad) + 1 (unk)
    OUTPUT_DIM = len(tgt_idx_to_char) # vocab size for Hindi : 64 + 1 (pad) + 1 (unk)
    EPOCHS = args.epochs

    print ('Input dimension is:', INPUT_DIM, 'Output dimension is:', OUTPUT_DIM)
    
    encoder = Encoder(INPUT_DIM, args.hidden_dim, args.embedding_size, args.num_layers, args.bidirectional, args.cell_type, args.dropout).to(device)
    decoder = Decoder(OUTPUT_DIM, args.hidden_dim, args.embedding_size, args.num_layers, args.bidirectional, args.cell_type, args.dropout).to(device)

    model = EncoderDecoder(encoder, decoder, args.cell_type, args.bidirectional, args.teacher_forcing_ratio, device).to(device)

    loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    elif args.optimizer == "nadam":
            optimizer= optim.NAdam(model.parameters(),lr=args.learning_rate)


    # Start the training
    prev_val_loss = 10000007
    for epoch in range(EPOCHS):
        print ('Started {epoch}...')
        epoch_loss = 0
        model.train()

        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):

            src = src.permute(1, 0) 
            tgt = tgt.permute(1, 0)  

            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, args.teacher_forcing_ratio)
            output = output[1:].reshape(-1, output.shape[2])

            tgt = tgt[1:].reshape(-1)
            # print ('Tgt shape', tgt.shape)
            # print ('output: ', output.shape)
            mask = (output!=1)
            output = output*mask

            # print ('What is going to the loss: ', tgt, " output: ", output)
            
            loss = loss_criterion(output, tgt)
            # print ('Loss shape: ', loss.shape)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += (loss.item())
            
            if batch_idx % 1000 == 0:
                print(f"Running epoch: {epoch}, batch: {batch_idx}")

        # Calculate word-level accuracy after every epoch
        val_loss, val_acc = calculate_accuracy(model,val_dataloader, tgt_idx_to_char,loss_criterion)
        
        if (val_loss < prev_val_loss):
            prev_val_loss = val_loss
            best_model_path = './ckpts/best_model_vanilla_encoder_decoder.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        # print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}")

        print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}, Val Acc: {val_acc}, Val loss: {val_loss}")
        # print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))})
        #wandb.log({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc,'train_acc': train_acc,'val_acc': val_acc})
    






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--cell_type', help="Choices:[RNN, LSTM, GRU]", type=str, default='LSTM')
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=128)
    parser.add_argument('-o', '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=2e-4)
    parser.add_argument('-em', '--embedding_size', help='size of embedding', type=int, default=512)
    parser.add_argument('-hd', '--hidden_dim', help='choices:[64,128,256,512]',type=int, default=512)
    parser.add_argument('-dp', '--dropout', help='choices:[0, 0.1, 0.2, 0.3]',type=float, default=0.2)
    parser.add_argument('-nl', '--num_layers', help='Number of layers in network: [1,3,5]',type=int, default=3)
    parser.add_argument('-bidir', '--bidirectional', help='Choices:["True","False"]',type=bool, default=False)
    parser.add_argument('-tf', '--teacher_forcing_ratio', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.7)
    parser.add_argument('-epc', '--epochs', help = 'Give the number of epochs - ideal between 20 to 30', default = 30)
    parser.add_argument('-pn' , '--project_name', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS6910_Assignment3_Q1')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='srija17199')


    

    args = parser.parse_args()
    train(args)
    # batch_size = args.batch_size
    # cell_type = args.cell_type
    # optimizer = args.optimizer
    # learning_rate = args.learning_rate
    # embedding_size = args.embedding_size
    # hidden_dim = args.hidden_dim
    # dropout = args.dropout
    # num_layers = args.num_layers
    # teacher_forcing_ratio = args.teacher_forcing_ratio
    # epochs = args.epochs
    # project_name = args.project_name
    # wandb_entity = args.wandb_entity
    
    if args.bidirectional.lower() == 'true':
        bidirectional = True
    elif args.bidirectional.lower() == 'false':
        bidirectional = False
    else:
        raise ValueError("Invalid value for --bidirectional. Use 'True' or 'False'.")

    # train(cell_type=args.cell_type,
    # batch_size=args.batch_size,
    # optimizer=args.optimizer,
    # learning_rate=args.learning_rate,
    # embedding_size=args.embedding_size,
    # hidden_dim=args.hidden_dim,
    # dropout=args.dropout,
    # num_layers=args.num_layers,
    # bidirectional=bidirectional,
    # teacher_forcing_ratio=args.teacher_forcing_ratio,
    # epochs=args.epochs,
    # project_name = args.project_name, 
    # wandb_entity = args.wandb_entity)

       

    

