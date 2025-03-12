import torch
from torch import nn
import ipdb
import torch.nn.functional as F 

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
    """_summary_

    Implementation of the BLSTM-based diarization model described in the paper:
    "End-to-end Neural Speaker Diarization with Permutation-free Objectives"
    by Fujita et al. (2019).

    [Paper](https://arxiv.org/abs/1909.05952)

                                                       
       Output                                          
         |                                             
         |                                             
+-----------------+                                    
| Linear + Sigmoid|          Label (One-hot conversion)
+--------|--------+                      |             
         |                    +----------|---------+   
  +------------+              |Deep Clustering Loss|   
  |    BLSTM   |              +----------|---------+   
  +------------+                         |             
         |                 +--------------------------+
         |-----------------|Linear + Tanh + Normalize |
  +------------+           +--------------------------+
  |    BLSTM   |                                       
  +------|-----+                                       
         |                                             
       Audio                                           
    
    """
    def __init__(self,
                n_speakers=20,
                dropout=0.25,
                in_size=80000, # 513 in the source code
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

        # LSTM for computing embeddings:
        self.bi_lstm_embed = nn.LSTM(input_size=in_size,
                                    hidden_size=hidden_size,
                                    num_layers= embedding_layers,
                                    batch_first=True,  
                                    bidirectional=True, 
                                    dropout=dropout)
        # LSTM for main branch:
        self.bi_lstm = nn.LSTM(input_size=hidden_size * 2,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)
        
        # Linear layer mapping LSTm1 output to speaker posterior probabilities.
        self.linear1 = nn.Linear(hidden_size * 2, n_speakers)
        
        # Linear layer mapping LSTM embedding output to embedding space.
        self.linear2 = nn.Linear(hidden_size * 2, embedding_size)

        self.dc_loss_ratio = dc_loss_ratio
        self.n_speakers = n_speakers

    def forward(self, x, hidden_state = None, activation=None):
        # x of shape [batch_size, 1, segment_length*sr]
        print(f"x.shape: {x.shape}")

        # Unpack hidden states
        if hidden_state is not None:
            hidden_state_in, cell_state_in, hidden_state_embed_in, cell_state_embed_in = hidden_state
        else:
            hidden_state_in, cell_state_in, hidden_state_embed_in, cell_state_embed_in = None, None, None, None

        # HACK: Why do we pass the initial hidden_states and not the output of the previous LSTM?
        if hidden_state is not None:
            embed, (hidden_state_embed, cell_state_embed) = self.bi_lstm_embed(x,
                                                                        (hidden_state_embed_in, cell_state_embed_in))

            y, (hidden_state, cell_state) = self.bi_lstm(embed,
                                                    (hidden_state_in, cell_state_in))
        else: 
            embed, (hidden_state_embed, cell_state_embed) = self.bi_lstm_embed(x)
            y, (hidden_state, cell_state) = self.bi_lstm(embed)

        
        # main branch
        # from [32,1,512] -> [32, 512]
        y_stack = y.view(-1, y.size(1) * y.size(2))
        y = self.linear1(y_stack)
        
        if activation is not None:
            y = activation(y)
        # what is this
        # irels = [xi.shape[1] for xi in x]
        # y = y.split(irels)

        # embedding branch
        embed_stack = embed.view(-1, embed.size(1) * embed.size(2))

        embed_out = torch.tanh(self.linear2(embed_stack))

        embed_out_normalized = F.normalize(embed_out, p=2, dim=1)

        return ((hidden_state, cell_state, hidden_state_embed, cell_state_embed),  y, embed_out_normalized)

