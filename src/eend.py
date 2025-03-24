# Copyright 2025 Barcelona Supercomputing Center (author: Marc Casals i Salvador)
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from typing import List, Tuple
from torch import nn
import ipdb
import logging
from feature_extractor import SpectrogramExtractor
import torch.nn.functional as F 

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion

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
                segment_length,
                frame_length,
                feature_extractor,
                sample_rate,
                feature_extractor_output_vectors_dimension,
                n_speakers=20,
                dropout=0.25,
                hidden_size=256,
                n_layers=1,
                embedding_layers=1,
                embedding_size=20,
                n_fft=2048,
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
        """
        super(BLSTM_EEND, self).__init__()
        self.segment_length = segment_length
        self.frame_length = frame_length
        self.n_speakers = n_speakers

        self.init_audio_feature_extractor(feature_extractor, sample_rate, feature_extractor_output_vectors_dimension)
        # LSTM for computing embeddings:
        self.bi_lstm_embed = nn.LSTM(input_size=feature_extractor_output_vectors_dimension,
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

        self.n_speakers = n_speakers

    def init_audio_feature_extractor(self, feature_extractor, sample_rate, feature_extractor_output_vectors_dimension):
        """
        This method initializes the audio feature extractor.

        There are two options:
        * SpectrogramExtractor
        * WavLMExtractor

        After this, it will be applied the Layer Normalization.

        .. math::
                y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

            The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
            is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
            is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
            the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
            :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
            :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
            The standard-deviation is calculated via the biased estimator, equivalent to
            `torch.var(input, unbiased=False)`.

        """
        if feature_extractor == 'SpectrogramExtractor':
            self.feature_extractor = SpectrogramExtractor(sample_rate,
                                                        feature_extractor_output_vectors_dimension)
        elif feature_extractor == 'WavLMExtractor':
            # TODO: Implement WavLMExtractor
            ...
        else:
            raise ValueError(f"Audio feature extractor {feature_extractor} not found")
        
        # Freeze all wavLM parameter except layers weights
        for name, parameter in self.feature_extractor.named_parameters():          
            if name != "layer_weights":
                logger.info(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False
        
        logger.debug(f"Feature extractor output vectors dimension: {feature_extractor_output_vectors_dimension}. Check if it suits the LayerNorm dimensions.")
        self.feature_extractor_norm_layer = nn.LayerNorm(feature_extractor_output_vectors_dimension)


    def forward(self, x:torch.Tensor, hidden_state: torch.Tensor = None, activation=None
                )->Tuple[Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        
        """The forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, 1, segment_length*sr]
            hidden_state (torch.Tensor, optional): The hidden state of the model. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]: _description_
        """

        # x of shape [batch_size, 1, segment_length*sr] -> [batch_size, segment_length*sr]
        x = x.squeeze(1)
        # x = x.view(x.size(0),int(self.segment_length/self.frame_length) , -1)

        # Extract Spectrogram Features from Audio. And Normalize them.
        x = self.feature_extractor(x)
        x = self.feature_extractor_norm_layer(x)
        # Features are of shape [batch_size, time, mel_bands].


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
        Batch, Time, Dimension = y.size()
        y_stack = y.contiguous().view(-1, Dimension)
        y_out = self.linear1(y_stack)
        y_out = y_out.view(Batch, Time, -1)
        
        if activation is not None:
            y_out = activation(y_out)
        # what is this
        # irels = [xi.shape[1] for xi in x]
        # y = y.split(irels)

        # embedding branch
        Batch, Time, Dimension = embed.size()
        embed_stack = embed.contiguous().view(-1, Dimension)
        embed_out = torch.tanh(self.linear2(embed_stack))
        embed_out_normalized = F.normalize(embed_out, p=2, dim=1)
        embed_out_normalized = embed_out_normalized.view(Batch, Time, -1)

        return ((hidden_state, cell_state, hidden_state_embed, cell_state_embed),  y_out, embed_out_normalized)

