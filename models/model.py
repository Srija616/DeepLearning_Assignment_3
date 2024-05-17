import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder module for sequence-to-sequence models.

    Args:
        input_dim (int): The size of the input vocabulary.
        hidden_dim (int): The number of features in the hidden state of the RNN/LSTM/GRU.
        num_layers (int): Number of recurrent layers.
        embedding_size (int): The size of the word embeddings.
        bidirectional (bool): If True, the RNN/LSTM/GRU layers will be bidirectional.
        cell_type (str): Type of recurrent cell to use. Options: 'RNN', 'LSTM', or 'GRU'.
        dp (float): Dropout probability to use in the RNN/LSTM/GRU layers.

    Raises:
        ValueError: If `cell_type` is not one of ['RNN', 'LSTM', 'GRU'].

    """


    def __init__(self, input_dim, hidden_dim,embedding_size, num_layers, bidirectional, cell_type, dp):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedded_size = embedding_size
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dp)
        self.direction = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.embedding_layer = nn.Embedding(input_dim,embedding_size)

        if cell_type == 'RNN':
              self.rnn = nn.RNN(self.embedding_size, self.hidden_dim, self.num_layers, dropout=dp,bidirectional=self.bidirectional)
        elif cell_type == 'LSTM':
              self.rnn = nn.LSTM(self.embedding_size, self.hidden_dim, self.num_layers, dropout=dp,bidirectional=self.bidirectional)
        elif cell_type == 'GRU':
              self.rnn = nn.GRU(self.embedding_size, self.hidden_dim, self.num_layers, dropout=dp,bidirectional=self.bidirectional)
        else:
            raise ValueError("Only valid cell types are: RNN, LSTM and GRU")     

    def forward(self, src):
        embedded_out = self.dropout(self.embedding_layer(src))
        # For cell type LSTM, returns the output and the hidden and the cell states in a single tuple
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded_out)
            return output, (hidden, cell)

        else:
            output, hidden = self.rnn(embedded_out) # For RNN or GRU returns the output and the hidden state
            return output,hidden

      
class Decoder(nn.Module):
    """
    Decoder module for sequence-to-sequence models.

    Args:
        output_dim (int): The size of the output vocabulary.
        hidden_dim (int): The number of features in the hidden state of the RNN/LSTM/GRU.
        embedding_size (int): The size of the word embeddings.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, the RNN/LSTM/GRU layers will be bidirectional.
        cell_type (str): Type of recurrent cell to use. Options: 'RNN', 'LSTM', or 'GRU'.
        dp (float): Dropout probability to use in the RNN/LSTM/GRU layers.

    Raises:
        ValueError: If `cell_type` is not one of ['RNN', 'LSTM', 'GRU'].
    """

    def __init__(self, output_dim, hidden_dim, embedding_size, num_layers, bidirectional, cell_type, dp):
        super(Decoder, self).__init__()
        # print ('Decoder Output_dim: ', output_dim)
        # print ('Decoder Hidden_dim: ', hidden_dim)
        # print ('Decoder Embedding_size: ', embedding_size)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedded_size=embedding_size     
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dp)
        self.direction = 2 if bidirectional else 1

        self.embedding_layer = nn.Embedding(output_dim,embedding_size)
        print ('output_dim: ', output_dim, hidden_dim, embedding_size)
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_dim, num_layers,dropout=dp)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_dim, num_layers,dropout=dp)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_dim, num_layers,dropout=dp)
        else:
            raise ValueError("Only valid cell types are: RNN, LSTM and GRU")
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, hidden):
        embedded_out = self.dropout(self.embedding_layer(input))
        output, hidden = self.rnn(embedded_out, hidden)
        output = self.fc_out(output)  # fully connected layer
        output = F.log_softmax(output, dim=1) # softmax for classification
        return output, hidden


class EncoderDecoder(nn.Module):
    """
    A sequence-to-sequence model with an encoder and a decoder, supporting bidirectional encoding 
    and optional teacher forcing during training.

    Args:
        encoder (nn.Module): The encoder module, which processes the input sequence.
        decoder (nn.Module): The decoder module, which generates the output sequence.
        cell_type (str): The type of RNN cell used ('RNN', 'LSTM', or 'GRU').
        bidirectional (bool): Whether the encoder is bidirectional.
        teacher_forcing_ratio (float): The probability of using teacher forcing during training.
        device (torch.device): The device on which the model is run (e.g., 'cpu' or 'cuda').

    Forward Args:
        src (Tensor): The source input sequence tensor of shape (src_len, batch_size).
        tgt (Tensor): The target output sequence tensor of shape (tgt_len, batch_size).
        teacher_forcing_ratio (float): The probability of using teacher forcing during this forward pass.

    Forward Returns:
        outputs (Tensor): The tensor containing the output sequences of shape (tgt_len, batch_size, tgt_vocab_size).
    """

    def __init__(self, encoder, decoder, cell_type, bidirectional, teacher_forcing_ratio, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type=cell_type
        self.bidirectional=bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio):
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(device)  #shape of the output is [max_len, batch_size, tgt_vocab_size]
        encoder_output, encoder_hidden = self.encoder(src)

        if self.bidirectional:
            if self.cell_type=='LSTM':
                hidden_states, cell_states = encoder_hidden
                forward_hidden = hidden_states[:self.encoder.num_layers, :, :]
                backward_hidden = hidden_states[self.encoder.num_layers:, :, :]

                forward_cell = cell_states[:self.encoder.num_layers, :, :]
                backward_cell = cell_states[self.encoder.num_layers:, :, :]

                avg_hidden = (forward_hidden + backward_hidden) / 2
                avg_cell = (forward_cell + backward_cell) / 2

                hidden_concat = (avg_hidden, avg_cell)

            else:
                hidden_concat = (encoder_hidden[0:self.encoder.num_layers,:,:] + encoder_hidden[self.encoder.num_layers:,:,:])/2
        else:
            hidden_concat = encoder_hidden
        
        decoder_hidden = hidden_concat
        # Initialize decoder input with the start token
        decoder_input = (tgt[0,:]).unsqueeze(0)

        for t in range(1,tgt.shape[0]):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            max_pr, idx = torch.max(decoder_output,dim=2)
            idx=idx.view(tgt.shape[1])

            # Determine the next decoder input using teacher forcing or predicted output
            teacher_force = teacher_forcing_ratio > torch.rand(1)
            if teacher_force:
                decoder_input= tgt[t,:].unsqueeze(0)
            else:
                decoder_input= idx.unsqueeze(0)

        return outputs