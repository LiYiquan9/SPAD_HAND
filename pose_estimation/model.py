from torch import nn
import torch
import math
import torch.nn.functional as F

class PositionalEncoder(torch.nn.Module):
    def __init__(self, device):
        super(PositionalEncoder, self).__init__()
        
        self.device = device
        
    def forward(self, params):
        """
        x: shape[batchsize*num_cameras,4]
        return: shape[batchsize*num_cameras,64]
        """
        params = params.to(self.device) 
        
        h = params[...,0]
        r = params[...,1]
        theta = params[...,2]
        phi = params[...,3]
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = h
        
        div_term = torch.exp(torch.arange(0, 16, 2).float() * (-math.log(10000.0) / 16)).to(self.device)
        
        pe_x = torch.zeros(params.size()[0], 16).to(self.device)
        pe_y = torch.zeros(params.size()[0], 16).to(self.device)
        pe_z = torch.zeros(params.size()[0], 16).to(self.device)
        pe_phi = torch.zeros(params.size()[0], 16).to(self.device)
        
        pe_x[:, 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe_x[:, 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        pe_y[:, 0::2] = torch.sin(y.unsqueeze(-1) * div_term)
        pe_y[:, 1::2] = torch.cos(y.unsqueeze(-1) * div_term)
        pe_z[:, 0::2] = torch.sin(z.unsqueeze(-1) * div_term)
        pe_z[:, 1::2] = torch.cos(z.unsqueeze(-1) * div_term)
        pe_phi[:, 0::2] = torch.sin(phi.unsqueeze(-1) * div_term)
        pe_phi[:, 1::2] = torch.cos(phi.unsqueeze(-1) * div_term)

        pe = torch.cat((pe_x, pe_y, pe_z, pe_phi), dim=-1)
        
        return pe
   
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
      
        N = values.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return self.fc2(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion * embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: shape[batchsize*num_cameras, 64]
        return: shape[batchsize*num_cameras, 64]
        """
        value = x
        key = x
        query = x
        attention = self.attention(value, key, query) # attention: shape[batchsize*num_cameras,64]
      
        # Add skip connection, followed by layer normalization
        x = self.norm1(attention + query)
        
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Add skip connection, followed by layer normalization
        return out

     
class MANOEstimator(nn.Module):
    
    def __init__(
        self,
        device,
        num_cameras=16,
        dropout_rate=0.1,
        
    ):
        super().__init__()
        
        self.device = device
        self.num_cameras = num_cameras

        self.layer0 = nn.Linear(64, 64)
        
        self.pe = PositionalEncoder(self.device)
        
        self.transformer = TransformerBlock(embed_size=64, heads=4, dropout=0.1, forward_expansion=1)
           
        # use a sequence of transformer blocks (optional)
        # self.transformers = nn.ModuleList([
        #     TransformerBlock(embed_size=64, heads=4, dropout=0.1, forward_expansion=1)
        #     for _ in range(3)  
        # ])
              
        self.layer1 = nn.Linear(64*self.num_cameras, 256) # concat
        
        self.layer2 = nn.Linear(256, 15*3 + 10)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bn0 = nn.BatchNorm1d(64) 
        
        self.bn1 = nn.BatchNorm1d(256)
        
        self.add_camera_weights = False
        
        self.camera_weights = nn.Parameter(torch.ones(num_cameras))
        
        
        
    def forward(self, x, h):
        """
        x: shape[batchsize, num_cameras, num_bins]
        h: shape[batchsize, num_cameras, num_params] # the postion params for each camera, now use 4
        return: shape[batchsize, 15*3 + 10]
        """
        x = x.to(self.device)
        h = h.to(self.device)
        
        batchsize, num_cameras, num_bins = x.size()
        num_params = h.size()[-1]

        x = x.view(batchsize * num_cameras, num_bins) # x: shape[batchsize * num_cameras, num_bins]
        h = h.view(batchsize * num_cameras, num_params)
        
        # hist_feature = self.conv1d(x).squeeze(1)
        hist_feature = self.layer0(x)
        
        hist_feature = F.relu((hist_feature))

        feature = hist_feature + self.pe(h) # shape[batchsize*num_cameras, 64]
            
        feature = feature.view(batchsize, num_cameras, num_bins)
        
        feature = self.transformer(feature)
        
        # use a sequence of transformer blocks (optional)
        # for transformer in self.transformers:
        #     feature = transformer(feature)

        concatenated_feature = feature.view(batchsize, -1) # concat
       
        feature = self.layer1(concatenated_feature)
        
        feature = F.relu((feature))
        
        feature = self.dropout(feature)
        
        output = self.layer2(feature) 
        
        return output
        
        
