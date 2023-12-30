import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    
    
    def __init__(
        self,
        in_channels,
        img_height,
        strip_width
        ):
        """
        Params:
            in_channels: the channels of the input spectrogram
            
            img_height: the height of the input spectrogram
            
            strip_width: the width of the strips
        """
        super().__init__()
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * img_height * strip_width,
            kernel_size=(img_height, strip_width),
            stride=strip_width
        )
        
        
    def forward(self, x):
        """
        Params:
            x: spectrogram (B, H, W, C)
            
        Returns:
            input embeddings
        """
        y = self.proj(x.permute(0,3, 1, 2)).flatten(2).transpose(1, 2)
        return y


class PositionalEmbedding(nn.Module):
    
    
    def __init__(
        self,
        img_width,
        strip_width,
        dim_feature,
        device
        ):
        """
        Params:
            img_width: the width of the input spectrogram
            
            strip_width: the width of the strips
            
            dim_feature: the feature dimension
        """
        super().__init__()
        
        self.seq_len = int(img_width / strip_width) + 1
        self.encoding = torch.zeros(self.seq_len, dim_feature, requires_grad=False)
        pos = torch.arange(0, self.seq_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, dim_feature, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim_feature)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim_feature)))
        
        
    def forward(self):
        """
        Returns:
            positional embeddings
        """
        return self.encoding[:self.seq_len, :]


class SpeechTransformer(nn.Module):
    
    
    def __init__(
        self,
        in_channels,
        img_height,
        img_width,
        strip_width,
        dim_feature,
        num_head,
        num_layers,
        dropout=0.1
        ):
        """
        Params:
            in_channels: the channels of the input spectrogram
            
            img_height: the height of the input spectrogram
            
            img_width: the width of the input spectrogram
            
            strip_width: the width of the strips
            
            dim_feature: the feature dimension
            
            num_head: the number of attention headss
            
            num_layers: the number of TransformerEncoderLayer
            
            dropout: the dropout ratio
        """
        super().__init__()
        
        self.input_embed = InputEmbedding(
            in_channels=in_channels,
            img_height=img_height,
            strip_width=strip_width
        )
        self.positional_embed = PositionalEmbedding(
            img_width=img_width,
            strip_width=strip_width,
            dim_feature=dim_feature
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_feature,
                nhead=num_head,
                dim_feedforward=2 * dim_feature,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
        )
        self.classier = nn.Sequential(
            nn.LayerNorm(dim_feature),
            nn.Linear(dim_feature, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=-1)
        )
        
        
    def forward(self, x):
        """
        Params:
            x: spectrogram (B, H, W, C)
            
        Returns:
            classes
        """
        x_embed = self.input_embed(x)
        B, _, C = x_embed.shape
        cls = torch.zeros(
            size=(B, 1, C),
            dtype=torch.float32,
            requires_grad=True
        ).to(x.device)
        x_embed = torch.concat([cls, x_embed], dim=1)
        pos_embed = self.positional_embed()
        feat = self.encoder(x_embed + pos_embed)
        cls_feat = feat[:, 0, :]
        classes = self.classier(cls_feat)
        
        return classes
    
    
    def weights_init(self, init_type='normal'):
        
        
        def init_func(module):
            classname = module.__class__.__name__
            if classname.find('Linear') == 0:
                if init_type == 'normal':
                    nn.init.normal_(module.weight.data, 0.0, 0.02)
                    
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight.data, gain=1.414)
                    
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight.data, gain=1.414)
                    
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                    
            elif classname.find('BatchNorm1d') ==0:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
                
        self.apply(init_func)   
