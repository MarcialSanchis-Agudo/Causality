"""
Script for the encoder and decoder layer for full transformer
"""

import torch 
import torch.nn.functional as F
from   self  import *
from    torch           import nn 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 activation="relu"):
        """
        The nn.Module for Feed-Forward network in transformer encoder/decoder layer 
        
        Args:
            d_model     :  (Int) The dimension of embedding 

            d_ff        :  (Int) The projection dimension in the FFD 
            
            activation  :  (Str) Activation function used in network

        """
        
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        if activation == "relu":
            self.act = nn.ReLU()
        if activation == "gelu":
            self.act = nn.GELU()
        if activation == "elu":
            self.act = nn.ELU()
            
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))



class selfEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, act_proj):
        """
        nn.Module for transformer Encoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            num_heads   :   (Int) The number of heads used in attention module
            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network 
            
            dropout     :   (Float) The dropout value to prevent from pverfitting

            act_proj    :   (Str)   The activation function used in the FFD
        """
        super(selfEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff,act_proj)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        The forward prop for the module 
        Args:
            x       :   Input sequence 
            
            mask    :   the mask used for attention, usually be the src_mask

        Returns:
            x       :   The encoded sequence in latent space       
        """
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x
    


class easyEncoderLayer(nn.Module):
    def __init__(self,attn, d_model, seqLen, num_heads, offset, d_ff, dropout, act_proj):
        """
        nn.Module for transformer Encoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            seqLen      :   (Int) The length of the input sequence
            
            num_heads   :   (Int) The number of heads used in attention module

            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network 
            
            dropout     :   (Float) The dropout value to prevent from pverfitting

            act_proj    :   (Str)   The activation function used in the FFD
        """
        super(easyEncoderLayer, self).__init__()
        if attn == 'easy':
            self.attn = EasyAttn(   d_model     =d_model, 
                                nmode       =seqLen, 
                                num_head    =num_heads,
                                offset      =offset)
        elif attn == 'cross':
            self.attn = Cross_SpaceTime_EasyAttn(   d_model     =d_model, 
                                seqLen       =seqLen, 
                                num_heads    =num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff,act_proj)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        The forward prop for the module 
        Args:
            x       :   Input sequence 
            
        Returns:
            x       :   The encoded sequence in latent space       
        """
        
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x
    
class Cross_SpaceTime_EasyAttn(nn.Module):
    
    def __init__(self, d_model, seqLen, num_heads):
        """
        Dense Easy attention mechansim used in transformer model for the time-series prediction and reconstruction
        Which do Split on WvR and Alpha both.
        
        Args:

            d_model     :   The embedding dimension for the input tensor 
            
            seqLen      :   The length of the sequence 

            num_head    :   The number of head to be used for multi-head attention
    
        """
        super(Cross_SpaceTime_EasyAttn,self).__init__()
     
        assert (d_model % num_heads == 0)  and (seqLen % num_heads == 0), "dmodel and seqLen must be divible by number of heads"

        self.d_model    =   d_model
        self.seqLen     =   seqLen
        self.d_k        =   d_model // num_heads
        self.d_t        =   seqLen // num_heads
        self.num_heads  =   num_heads
        # Create the tensors
        self.Alpha      = nn.Parameter(torch.randn(size=(num_heads,seqLen,seqLen)    ,       dtype=torch.float),requires_grad=True)            
        # The Left matrix for sloving the time scale
        self.WVL         = nn.Parameter(torch.randn(size=(seqLen,seqLen)            ,        dtype=torch.float),requires_grad=True)               
        # The Right Matrix for solving the space scale
        self.WVR         = nn.Parameter(torch.randn(size=(num_heads, d_model,d_model) ,  dtype=torch.float),requires_grad=True)               
        # Initialisation
        nn.init.xavier_uniform_(self.Alpha)
        nn.init.xavier_uniform_(self.WVL)
        nn.init.xavier_uniform_(self.WVR)
        self.split_probs  =   nn.Parameter(torch.ones(seqLen, dtype=torch.float)/seqLen) 
        self.count = 0
        self.total = 300
        self.best_split_probs = None
        self.temperature = 1.0 
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        #self.split_probs.to('cuda:0').float
    
    def split_on_space(self, x):
        """
        Split the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S, N//H]
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_on_space(self, x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        batch_size, _, seq_length, d_model = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    

    def split_on_time(self,x):
        """
        Split the sequence into multi-heads

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S//H, N]
        
        """
        
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, self.num_heads, self.d_t, d_model)
    
    def combine_on_time(self,x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        batch_size, _, d_t, d_model = x.size()
        return x.contiguous().view(batch_size, self.seqLen, d_model)

 
    def forward(self,x:torch.Tensor):   
        """
        Forward prop for the easyattention module 
        
        Following the expression:  x_hat    =   Alpha @ Wv @ x 

        Args:  
            
            self    :   The self objects

            x       :   A tensor of Input data
        
        Returns:
            
            x       :   The tensor be encoded by the moudle
        
        """
        # Obtain the value of batch size 
        B,_,_ =   x.shape
        #print(1,x.shape)
        # Start from WvL 
        x_    = self.self_attn(x, x, x, mask=None)
        x     =   torch.bmm ( self.WVL.repeat(B,1,1), x )
        #print(2,x.shape)
        # We split on the time dimension and apply multihead on Right 
        x     =   self.split_on_time(x) 
        #print(3,x.shape)
        x     =   x @ self.WVR.repeat(B,1,1,1)

        x     =   self.combine_on_time(x)
        #print(4,x.shape)
        # We split on the space dimension as self-attention, and apply attion score
        x     =   self.split_on_space(x)

        x     =   self.Alpha.repeat(B,1,1,1) @ x
        #print(5,x.shape)
        x     =   self.combine_on_space(x) + x_

        return x
