import torch
from torch import nn
import ipdb 

class EEND_Model(nn.Module):
    def __init__(self, params, device) -> None:
        super().__init__()
        self.device = device
        # Instantiate the linear layer here so it becomes a registered parameter.
        # Assuming you know the input dimension (for example, params.input_dim)
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        print("we print x:size", x.size())
        out = self.linear(x)
        return out
    
class BLSTM_EEND(nn.Module):
    def __init__(self,
                n_speakers=4,
                dropout=0.25,
                in_size=513,
                hidden_size=256,
                n_layers=1,
                embedding_layers=1,
                embedding_size=20,
                dc_loss_ratio=0.5,
                ):
        """ BLSTM-based diarization model.

        Args:
            n_speakers (int): Number of speakers in recording
            dropout (float): dropout ratio
            in_size (int): Dimension of input feature vector
            hidden_size (int): Number of hidden units in LSTM
            n_layers (int): Number of LSTM layers after embedding
            embedding_layers (int): Number of LSTM layers for embedding
            embedding_size (int): Dimension of embedding vector
            dc_loss_ratio (float): mixing parameter for DPCL loss
        """
        super(BLSTM_EEND, self).__init__()

        self.bi_lstm = nn.LSTM(hidden_size, hidden_size * 2, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.bi_lstm_embed = nn.LSTM(in_size, embedding_size, embedding_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.linear1 = nn.Linear(embedding_size * 2, n_speakers)
        self.linear2 = nn.Linear(embedding_size * 2, embedding_size)

        self.dc_loss_ratio = dc_loss_ratio
        self.n_speakers = n_speakers

    def forward(self, x, hidden_state = None, activation=None):
        print(f"x.shape: {x.shape}")
        # Unpack hidden states
        if hidden_state is not None:
            hidden_state_in, cell_state_in, hidden_state_embed_in, cell_state_embed_in = hidden_state
            # forward to LSTM layers
            embed, hidden_state_embed, cell_state_embed = self.bi_lstm_embed(x,
                                                                        hidden_state_embed_in, 
                                                                        cell_state_embed_in)            
        else:
            hidden_state_in, cell_state_in, hidden_state_embed_in, cell_state_embed_in = None, None, None, None
            # forward to LSTM layers
            embed, hidden_state_embed, cell_state_embed = self.bi_lstm_embed(x)        

        print(f"embed.shape: {embed.shape}")
        print(f"hidden_state_embed.shape: {hidden_state_embed.shape}")
        print(f"cell_state_embed.shape: {cell_state_embed.shape}")
        # HACK: Why do we pass the initial hidden_states and not the output of the previous LSTM?
        y, hidden_state, cell_state = self.bi_lstm(embed,
                                                hidden_state_in, 
                                                cell_state_in)
        print(f"y.shape: {y.shape}")
        print(f"hidden_state.shape: {hidden_state.shape}")
        print(f"cell_state.shape: {cell_state.shape}")
        
        # main branch
        y_stack = y.view(-1, y.size(1) * y.size(2))
        y = self.linear1(y_stack)

        if activation is not None:
            y = activation(y)
        irels = [xi.shape[1] for xi in x]
        y = y.split(irels)

        # embedding branch
        embed_stack = embed.view(-1, embed.size(1) * embed.size(2))
