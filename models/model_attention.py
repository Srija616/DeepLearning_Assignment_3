import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_context_vector(context_vector, step):
    plt.figure(figsize=(10, 8))
    plt.imshow(context_vector.cpu().detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Context Vector")
    plt.xlabel("Time Steps")
    plt.ylabel("Features")
    
    # Save plot to a file
    plt.savefig(f"context_vector_step_{step}.png")
    plt.close()
    
    # Log the plot to wandb
    wandb.log({"context_vector": wandb.Image(f"context_vector_step_{step}.png")}, step=step)


class Encoder(nn.Module):
    """
    Encoder module for sequence-to-sequence models.

    Args:
        input_dim (int): The size of the input vocabulary.
        hidden_dim (int): The number of features in the hidden state of the RNN/GRU.
        num_layers (int): Number of recurrent layers.
        embedding_size (int): The size of the word embeddings.
        bidirectional (bool): If True, the RNN/LSTM/GRU layers will be bidirectional.
        cell_type (str): Type of recurrent cell to use. Options: 'RNN', 'LSTM', or 'GRU'.
        dp (float): Dropout probability to use in the RNN/LSTM/GRU layers.

    Raises:
        ValueError: If `cell_type` is not one of ['RNN', 'LSTM', 'GRU'].

    """


    def __init__(self, input_dim, hidden_dim,embedding_size,cell_type, dp):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dp)
        
        self.embedding_layer = nn.Embedding(input_dim,embedding_size)

        if cell_type == 'RNN':
              self.rnn = nn.RNN(self.embedding_size, self.hidden_dim, self.num_layers)
        elif cell_type == 'GRU':
              self.rnn = nn.GRU(self.embedding_size, self.hidden_dim, self.num_layers)
        else:
            raise ValueError("Only valid cell types are: RNN and GRU")     

    def forward(self, src):

        embedded_out = self.dropout(self.embedding_layer(src))
        output, hidden = self.rnn(embedded_out)
        return output, hidden

      
class Decoder(nn.Module):
    """
    Decoder module for sequence-to-sequence models.

    Args:
        output_dim (int): The size of the output vocabulary.
        hidden_dim (int): The number of features in the hidden state of the RNN/LSTM/GRU.
        embedding_size (int): The size of the word embeddings.
        num_layers (int): Number of recurrent layers.
        cell_type (str): Type of recurrent cell to use. Options: 'RNN', 'LSTM', or 'GRU'.
        dp (float): Dropout probability to use in the RNN/LSTM/GRU layers.

    Raises:
        ValueError: If `cell_type` is not one of ['RNN' and 'GRU'].
    """

    def __init__(self, output_dim, hidden_dim, embedding_size, cell_type, dp):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding_size=embedding_size     
        self.cell_type = cell_type
        self.num_layers = 1
        self.dropout = nn.Dropout(dp)
        self.embedding_layer = nn.Embedding(output_dim,embedding_size)

        if cell_type == 'RNN':
            self.rnn = nn.RNN(self.hidden_dim + self.embedding_size, self.hidden_dim, self.num_layers)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(self.hidden_dim + self.embedding_size, self.hidden_dim, self.num_layers)
        else:
            raise ValueError("Only valid cell types are: RNN, LSTM and GRU")
        
        self.fc1 = nn.Linear(self.hidden_dim, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, self.output_dim, bias = False)
        self.e = nn.Linear((2*self.hidden_dim),1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.fc.weight = self.embedding_layer.weight

    def forward(self, input, encoder_states, hidden):
        input = input.unsqueeze(0)
        embedded_out = self.dropout(self.embedding_layer(input)).squeeze(0)
        hidden_repeated = hidden[0].repeat(encoder_states.shape[0], 1, 1)  # Match hidden state to sequence length
        

        combined_states = torch.cat((hidden_repeated, encoder_states), dim=2)
        relu_out = self.relu(self.e(combined_states))
        attention_weights = self.softmax(relu_out).transpose(0, 1).transpose(1, 2)
        
        # After attention calculation, it needs to be multiplied with the encoder states to get the context vectors
        encoder_states = encoder_states.transpose(0, 1)
        context_vector = torch.bmm(attention_weights, encoder_states).transpose(0,1)

        if not self.training:
            plot_context_vector(context_vector[0], step=0)

        rnn_input = torch.cat((context_vector, embedded_out), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        fc_1 = self.fc1(output)
        output = self.fc(fc_1).squeeze(0)
        return output, hidden
            
        


class EncoderDecoder(nn.Module):
    """
    A sequence-to-sequence model with an encoder and a decoder, supporting optional 
    teacher forcing during training.

    Args:
        encoder (nn.Module): The encoder module, which processes the input sequence.
        decoder (nn.Module): The decoder module, which generates the output sequence.
        cell_type (str): The type of RNN cell used ('RNN', 'LSTM', or 'GRU').
        teacher_forcing_ratio (float): The probability of using teacher forcing during training.
        device (torch.device): The device on which the model is run (e.g., 'cpu' or 'cuda').

    Forward Args:
        src (Tensor): The source input sequence tensor of shape (src_len, batch_size).
        tgt (Tensor): The target output sequence tensor of shape (tgt_len, batch_size).
        teacher_forcing_ratio (float): The probability of using teacher forcing during this forward pass.

    Forward Returns:
        outputs (Tensor): The tensor containing the output sequences of shape (tgt_len, batch_size, tgt_vocab_size).
    """
    
    def __init__(self, encoder, decoder, cell_type, teacher_forcing_ratio, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type=cell_type
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio):
        batch_size, max_len = tgt.shape[1], tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim
        
        
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(device)  #shape of the output is [max_len, batch_size, tgt_vocab_size]
        encoder_output, encoder_hidden = self.encoder(src) # for LSTM the encoder_hidden would be a tuple
        decoder_hidden = encoder_hidden
        decoder_input = (tgt[0,:]).unsqueeze(0)

        for t in range(1,max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_output, decoder_hidden)
            outputs[t] = decoder_output
            max_prob, idx = torch.max(decoder_output,dim=1)
            
            idx=idx.view(batch_size)
            teacher_force = teacher_forcing_ratio > torch.rand(1)
            if teacher_force:
                decoder_input= tgt[t,:].unsqueeze(0)
            else:
                decoder_input= idx.unsqueeze(0)
        
        return outputs