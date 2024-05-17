import wandb
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
from train import dataloader
from model import Encoder, Decoder, EncoderDecoder
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")






# Total possible runs = 3 * 3 * 2 * 5 * 4 * 3 * 4 * 2 * 2 * 3
sweep_config= {
"name" : "DeepLearning_Assignment3",
"method" : "bayes",
'metric': {
    'name': 'val_acc',
    'goal': 'maximize'
},
'parameters' : {
    'cell_type' : { 'values' : ['RNN','LSTM','GRU'] },
    'batch_size' : {'values' : [32,64,128]},
    'optimizer':{"values": ['adam','nadam']},
    'learning_rate':{"values": [0.005, 0.0002, 0.0005]},
    'embedding_size' : {'values' : [64,128,256,512]},
    'hidden_dim' : {'values' : [128,256,512]},
    'dropout' : { 'values' : [0,0.1,0.2,0.5]},
    'num_layers' : {'values' : [1,3]},
    'bidirectional' : {'values' : [True ,False]},
    'teacher_forcing_ratio':{"values":[0.2, 0.7]},
    }
}

folder = './data/hin/'
train_csv = f'{folder}/hin_train.csv'
test_csv = f'{folder}/hin_test.csv'
valid_csv = f'{folder}/hin_valid.csv'



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






def train_sweep():
    wandb.init(project='DL-Assignment3')
    config= wandb.config
    name = "celltype_" + str(config.cell_type) + "_bs_" + str(config.batch_size) + "_optim_" + str(config.optimizer) +"_lr_" + str(config.learning_rate) 
    wandb.run.name=name

    print ('Started dataloading...')
    
    train_dataloader, val_dataloader, test_dataloader, src_idx_to_char, tgt_idx_to_char = dataloader(train_csv, valid_csv, test_csv, config.batch_size)
    
    INPUT_DIM = len(src_idx_to_char)  # vocab size for English : 26 + 1 (pad) + 1 (unk)
    OUTPUT_DIM = len(tgt_idx_to_char) # vocab size for Hindi : 64 + 1 (pad) + 1 (unk)
    EPOCHS = 30

    print ('Input dimension is:', INPUT_DIM, 'Output dimension is:', OUTPUT_DIM)
    
    encoder = Encoder(INPUT_DIM, config.hidden_dim, config.embedding_size, config.num_layers, config.bidirectional, config.cell_type, config.dropout).to(device)
    decoder = Decoder(OUTPUT_DIM, config.hidden_dim, config.embedding_size, config.num_layers, config.bidirectional, config.cell_type, config.dropout).to(device)

    model = EncoderDecoder(encoder, decoder, config.cell_type, config.bidirectional, config.teacher_forcing_ratio, device).to(device)

    loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)
    elif config.optimizer == "nadam":
            optimizer= optim.NAdam(model.parameters(),lr=config.learning_rate)


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
            
            output = model(src, tgt, config.teacher_forcing_ratio)
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
        test_loss, test_acc = calculate_accuracy(model,test_dataloader, tgt_idx_to_char,  loss_criterion)
        
        if (val_loss < prev_val_loss):
            prev_val_loss = val_loss
            best_model_path = f'./ckpts/sweeps/vanilla_encoder_decoder/{wandb.run.name}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        # print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}")

        print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}, Val Acc: {val_acc}, Val loss: {val_loss}")

        wandb.log({'epoch': EPOCHS, 'train_loss': loss.item(), 'test_acc': test_acc,'val_acc': val_acc,'test_loss': test_loss,'val_loss': val_loss})
        # Save the best model
    wandb.run.save()
    wandb.run.finish()
    return
        

# import os
# os.system('wandb login --relogin')
sweep_id = wandb.sweep(sweep_config, project="DL-Assignment3", entity="srija17199")
wandb.agent(sweep_id, function=train_sweep,count=20, project='DL-Assignment3')
