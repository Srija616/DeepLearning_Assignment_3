import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import dataloader, calculate_accuracy
from model_attention import Encoder, Decoder, EncoderDecoder
from tqdm import tqdm

print ('Execution started')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

folder = './data/hin/'
train_csv = f'{folder}/hin_train.csv'
test_csv = f'{folder}/hin_test.csv'
valid_csv = f'{folder}/hin_valid.csv'



def train(args):
    print ('Started dataloading...')
    
    train_dataloader, val_dataloader, test_dataloader, src_idx_to_char, tgt_idx_to_char = dataloader(train_csv, valid_csv, test_csv, args.batch_size)
    
    input_dim = len(src_idx_to_char)  # vocab size for English : 26 + 1 (pad) + 1 (unk)
    output_dim = len(tgt_idx_to_char) # vocab size for Hindi : 64 + 1 (pad) + 1 (unk)
    epochs = args.epochs

    print ('Input dimension is:', input_dim, 'Output dimension is:', output_dim)
    
    encoder = Encoder(input_dim, args.hidden_dim, args.embedding_size, args.cell_type, args.dropout).to(device)
    decoder = Decoder(output_dim, args.hidden_dim, args.embedding_size, args.cell_type, args.dropout).to(device)

    model = EncoderDecoder(encoder, decoder, args.cell_type, args.teacher_forcing_ratio, device).to(device)

    loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    elif args.optimizer == "nadam":
            optimizer= optim.NAdam(model.parameters(),lr=args.learning_rate)


    # Start the training
    prev_val_loss = 10000007
    prev_test_acc = 0
    for epoch in range(epochs):
        print ('Started {epoch}...')
        epoch_loss = 0
            

        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):

            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, args.teacher_forcing_ratio)
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
            # best_model_path = './ckpts_attention/indeed_the_best_attention_model.pth'
            # torch.save(model.state_dict(), best_model_path)
            # print(f"Best model saved to {best_model_path}")

        if (test_acc> prev_test_acc):
            df = pd.DataFrame(predictions)
            csv_file = f"./predictions_attention/{test_acc}.csv"
            df.to_csv(csv_file, index=False, header=False)


        print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_dataloader))}, Val Acc: {val_acc}, Val loss: {val_loss}, Test Acc: {test_acc}, Test Loss: {test_loss}")

    model.eval()
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(tqdm(test_dataloader)):
            src = src[:10]
            tgt = tgt[:10]
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)
            output = model(src, tgt, 0)
            break



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--cell_type', help="Choices:[RNN, GRU]", type=str, default='GRU')
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=64)
    parser.add_argument('-o', '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0005)
    parser.add_argument('-em', '--embedding_size', help='size of embedding', type=int, default=64)
    parser.add_argument('-hd', '--hidden_dim', help='choices:[64,128,256,512]',type=int, default=256)
    parser.add_argument('-dp', '--dropout', help='choices:[0, 0.1, 0.2, 0.3]',type=float, default=0.5)
    parser.add_argument('-tf', '--teacher_forcing_ratio', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.5)
    parser.add_argument('-epc', '--epochs', help = 'Give the number of epochs - ideal between 20 to 30', default = 30)
    parser.add_argument('-pn' , '--project_name', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS6910_Assignment3_Q1')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='srija17199')

    args = parser.parse_args()
    train(args)
       

    

