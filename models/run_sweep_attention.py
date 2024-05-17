import pandas as pd
import os
import argparse
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import TransliterationDataLoader, Vocab, dataloader, calculate_accuracy, get_word_from_index
from model_attention import Encoder, Decoder, EncoderDecoder
from tqdm import tqdm

print ('Execution started')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

# Data paths
folder = './data/hin/'
train_csv = f'{folder}/hin_train.csv'
test_csv = f'{folder}/hin_test.csv'
valid_csv = f'{folder}/hin_valid.csv'

sweep_config= {
"name" : "DeepLearning_Assignment3-Attention",
"method" : "bayes",
'metric': {
    'name': 'val_acc',
    'goal': 'maximize'
},
'parameters' : {
    'cell_type' : { 'values' : ['RNN','GRU'] },
    'batch_size' : {'values' : [32,64]},
    'optimizer':{"values": ['adam','nadam']},
    'learning_rate':{"values": [0.0002, 0.0005, 0.0007]},
    'embedding_size' : {'values' : [64,128,256,512]},
    'hidden_dim' : {'values' : [128,256,512]},
    'dropout' : { 'values' : [0.2,0.5]},
    'teacher_forcing_ratio':{"values":[0.5, 0.7]},
    }
}

# Run a sweep
def train_sweep():
    wandb.init(project='DL-Assignment3')
    config= wandb.config
    name = "celltype_" + str(config.cell_type) + "_bs_" + str(config.batch_size) + "_optim_" + str(config.optimizer) +"_lr_" + str(config.learning_rate) 
    wandb.run.name=name
    
    # make the folder to save test output csv
    folder_path = f'./predictions_attention/{name}'
    os.makedirs(folder_path, exist_ok = True)
    print ('Started dataloading...')
    
    train_dataloader, val_dataloader, test_dataloader, src_idx_to_char, tgt_idx_to_char = dataloader(train_csv, valid_csv, test_csv, config.batch_size)
    
    input_dim = len(src_idx_to_char)  # vocab size for English : 26 + 1 (pad) + 1 (unk)
    output_dim = len(tgt_idx_to_char) # vocab size for Hindi : 64 + 1 (pad) + 1 (unk)
    epochs = 30  # run only for 30 epochs

    print ('Input dimension is:', input_dim, 'Output dimension is:', output_dim)
    
    encoder = Encoder(input_dim, config.hidden_dim, config.embedding_size, config.cell_type, config.dropout).to(device)
    decoder = Decoder(output_dim, config.hidden_dim, config.embedding_size, config.cell_type, config.dropout).to(device)
    model = EncoderDecoder(encoder, decoder, config.cell_type, config.teacher_forcing_ratio, device).to(device)

    loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)
    elif config.optimizer == "nadam":
            optimizer= optim.NAdam(model.parameters(),lr=config.learning_rate)


    # Start the training
    prev_val_loss = 10000007
    prev_test_acc = 0
    for epoch in range(epochs):
        print ('Started {epoch}...')
        epoch_loss = 0
        model.train()

        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):

            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, config.teacher_forcing_ratio)
            output = output[1:].reshape(-1, output.shape[2])

            tgt = tgt[1:].reshape(-1)
            mask = (output!=1)
            output = output*mask

            # print ('What is going to the loss: ', tgt, " output: ", output)
            
            loss = loss_criterion(output, tgt)
            # print ('Loss shape: ', loss.shape)
            loss.backward()
            optimizer.step()
            epoch_loss += (loss.item())

        # Calculate word-level accuracy after every epoch
        val_loss, val_acc, predictions_val = calculate_accuracy(model,val_dataloader, tgt_idx_to_char,loss_criterion)
        test_loss, test_acc, predictions = calculate_accuracy(model,test_dataloader, tgt_idx_to_char,  loss_criterion, log_to_file = True)
        
        if (val_loss < prev_val_loss):
            prev_val_loss = val_loss

        if (test_acc> prev_test_acc):
            df = pd.DataFrame(predictions)
            csv_file = f"./predictions_attention/{name}/predictions.csv"
            df.to_csv(csv_file, index=False, header=False)


        print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}, Val Acc: {val_acc}, Val loss: {val_loss}, Test Acc: {test_acc}, Test Loss: {test_loss}")
        
        wandb.log({'epoch': epochs, 'train_loss': loss.item(), 'test_acc': test_acc,'val_acc': val_acc,'test_loss': test_loss,'val_loss': val_loss})
    wandb.run.save()
    wandb.run.finish()
    return

import os
os.system('wandb login --relogin')
sweep_id = wandb.sweep(sweep_config, project='DL-Assignment3', entity="srija17199")
wandb.agent(sweep_id, function=train_sweep,count=15, project='DL-Assignment3', entity="srija17199")


    

