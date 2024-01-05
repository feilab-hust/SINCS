import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from siren_pytorch import SirenNet, SirenWrapper

from core.models.Embedder import *
from core.utils.Coord import Coord
from glbSettings import *


class Model:
    def __init__(self, model_type:str, *args, **kwargs):
        self.type = model_type
        self.model = self.get_by_name(model_type, *args, **kwargs)

    def get_by_name(self, model_type:str, *args, **kwargs):
        try:
            model = eval(model_type)(*args, **kwargs)
        except:
            raise ValueError(f"Type {model_type} not recognized!")
        return model

    def eval(self, x, embedder, post, model_kwargs):
        embedder_kwargs = model_kwargs['embedder']
        post_kwargs = model_kwargs['post']
        h = x
        h = embedder.embed(h, **embedder_kwargs)
        h = self.model(h)
        h = post(h, **post_kwargs)
        return h
    
    def get_model(self):
        return self.model

    def get_state(self):
        return self.model.get_state()

    def load(self, ckpt):
        self.model.load_params(ckpt)

    def eval1(self, x, t,post, model_kwargs):
        post_kwargs = model_kwargs['post']
        embed_kwargs = model_kwargs['embed_fn']
        embedtime_kwargs = model_kwargs['embedtime_fn']
        h = self.model(x,t,**embed_kwargs,**embedtime_kwargs)
        h = post(h, **post_kwargs)
        return h

class BasicModel(nn.Module):
    """Basic template for creating models."""
    def __init__(self):
        super().__init__()

    def forward(self):
        """To be overwrited"""
        raise NotImplementedError
    
    def load_params(self, path):
        """To be overwrited"""
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError


class NeRF(BasicModel):
    """Standard NeRF model"""
    def __init__(self, D=8, W=256, input_ch=3,output_ch=1,input_ch_time=1, skips=[4],embed_fn=None,zero_canonical=True, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips


        layers = [nn.Linear(input_ch, W)]
        for i in range(D-1):
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers.append(nn.Linear(in_channels, W))
        self.pts_linear = nn.ModuleList(layers)
        self.output_linear = nn.Linear(W, output_ch)
        self.to(DEVICE)
        
    def forward(self, x):
        h = x
        for i,l in enumerate(self.pts_linear):
            h = l(h)
            h = F.relu(h)
            # h = F.leaky_relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

        outputs = self.output_linear(h)
        return outputs


    
    def load_params(self, ckpt:dict):
        # Load model
        self.load_state_dict(ckpt['network_fn_state_dict'])

    def get_state(self):
        return self.state_dict()




class Siren(BasicModel):
    """Siren model"""
    def __init__(self, dim_in=3, dim_hidden=256, dim_out=1, num_layers=3, w0_initial=30., *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # self.hash_table = hash_table

        self.net = SirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            w0_initial=w0_initial
        )
        self.to(DEVICE)

    def forward(self, x):
        self.output_activation = nn.ReLU()
        # mapped_coordinates = self.hash_table(x)
        # h = mapped_coordinates
        out = self.net(x)
        # out = self.output_activation(out)
        return out

    def load_params(self, ckpt: dict):
        # Load model
        self.load_state_dict(ckpt['network_fn_state_dict'])

    def get_state(self):
        return self.state_dict()


def create_model(Flags, shape, model_type='NeRF', embedder_type='PositionalEncoder', post_processor_type='relu', lr=1e-4, weights_path=None):
    ## embedder
    embedder_kwargs = {
        # PositionalEmbedder
        'multires': Flags.multires, 
        'multiresZ': Flags.multiresZ,
        'multiresT': Flags.multiresT,
        'input_dim': Flags.input_dim
    }
    embedder = Embedder.get_by_name(embedder_type, **embedder_kwargs)
    ## model
    model_kwargs={
                # NeRF
                'D': Flags.netdepth,
                'W': Flags.netwidth,
                'input_ch': embedder.out_dim if hasattr(embedder, 'out_dim') else 4,
                'output_ch': Flags.sigch,
                'skips': Flags.skips, 



    }
    model = Model(model_type=model_type, **model_kwargs)
    grad_vars = list(model.get_model().parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lr, betas=(0.9,0.999), capturable=(DEVICE.type == 'cuda'))
    start = 0
    # Load checkpoint
    if weights_path != None :
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load(ckpt)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt['global_step']


    ## post processor
    if post_processor_type == 'linear':
        post_processor = lambda x:x
    elif post_processor_type == 'relu':
        post_processor = F.relu
    elif post_processor_type == 'leakrelu':
        post_processor = F.leaky_relu
    else:
        raise ValueError(f"Type {post_processor_type} not recognized!")
    return model, embedder, post_processor,optimizer, start


class Embedder1:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder1(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def create_siren_model(Flags, shape, model_type='Siren', lr=1e-4, weights_path=None):
    ## model
    model_kwargs={
                #Siren
                'num_layers': Flags.netdepth,
                'dim_hidden': Flags.netwidth,
                'dim_in': 3,
                'dim_out': Flags.sigch,
                'w0_initial': Flags.w0,

    }
    model = Siren(**model_kwargs)
    grad_vars = list(model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lr, betas=(0.9,0.999), capturable=(DEVICE.type == 'cuda'))
    start = 0
    # Load checkpoint
    if weights_path != None:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load(ckpt)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt['global_step']
    return model, optimizer, start
if __name__ == '__main__':
    model = NeRF()
    print(model.parameters)