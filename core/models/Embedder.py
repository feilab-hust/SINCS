"""
Create positional encoding embedder
"""
import torch

from glbSettings import *
import numpy as np

class Embedder:
    @staticmethod
    def get_by_name(embedder_type:str=None, *args, **kwargs):
        if embedder_type is None or embedder_type == 'None':
            embedder_type = "BasicEmbedder"
        try:
            embedder = eval(embedder_type)(*args, **kwargs)
        except:
            raise ValueError(f"Type {embedder_type} not recognized!")
        return embedder

class BasicEmbedder:
    """
    An embedder that do nothing to the input tensor.
    Used as a template.
    """
    def __init__(self, *args, **kwargs):
        """To be overwrited"""
        del args, kwargs
    
    def embed(self, inputs):
        """To be overwrited"""
        return inputs

class PositionalEncoder(BasicEmbedder):
    def __init__(self, multires=0, multiresZ = None, multiresT=None,input_dim=3, embed_bases=[torch.sin,torch.cos], include_input=True, *args, **kwargs):
        del args, kwargs
        self.multires = multires
        if input_dim > 2 and multiresZ is not None:
            self.multiresZ = multiresZ
        if input_dim > 3 and multiresT is not None:
            self.multiresT = multiresT
        self.in_dim = input_dim
        self.embed_bases = embed_bases
        self.include_input = include_input
        # self.include_input=False
        self.embed_fns = []
        self.out_dim = 0
        self._create_embed_fn()
        if multiresT is not None:
            self.embed = self._embed_iso if not hasattr(self, 'embed_fns_T') and not hasattr(self, 'embed_fns_Z') else self._embed_aniso1
        if multiresT is None:
            self.embed = self._embed_iso if (not hasattr(self, 'embed_fns_Z')) else self._embed_aniso



    def _create_embed_fn(self):
        embed_fns = []
        d = (self.in_dim - 2) if (hasattr(self, 'multiresT') and hasattr(self, 'multiresZ')) else \
            (self.in_dim - 1) if (hasattr(self, 'multiresZ') and not hasattr(self, 'multiresT')) else \
                self.in_dim
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.multires-1
        N_freqs = self.multires
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs).to(DEVICE)

        #radia pe
        dia_digree=30
        s = np.sin(np.arange(0, 180, dia_digree) * np.pi / 180)[
            :, np.newaxis
            ]
        c = np.cos(np.arange(0, 180, dia_digree) * np.pi / 180)[
            :, np.newaxis
            ]
        fourier_mapping = np.concatenate((s, c), axis=1).T
        fourier_mapping = torch.from_numpy(np.float32(fourier_mapping)).to(DEVICE)
        radia_dim = len(s)
        # xy_freq = tf.matmul(in_node[:, :2], fourier_mapping)

        for freq in freq_bands:
            for p_fn in self.embed_bases:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * torch.pi*freq))
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(torch.matmul(x,fourier_mapping) * torch.pi * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x*torch.pi* freq))
                out_dim += radia_dim

        if hasattr(self, 'multiresZ'):
            embed_fns_Z = [] 
            d = 1
            if self.include_input:
                embed_fns_Z.append(lambda x : x)
                out_dim += d
            max_freq_Z = self.multiresZ-1
            N_freqs_Z = self.multiresZ
            freq_bands_Z = 2.**torch.linspace(0., max_freq_Z, steps=N_freqs_Z).to(DEVICE)

            for freqZ in freq_bands_Z:
                for p_fn in self.embed_bases:
                    embed_fns_Z.append(lambda x, p_fn=p_fn, freq=freqZ : p_fn(x *torch.pi* freq))
                    out_dim += d
            self.embed_fns_Z = embed_fns_Z

        if hasattr(self, 'multiresT'):
            embed_fns_T = []
            d = 1
            if self.include_input:
                embed_fns_T.append(lambda x: x)
                out_dim += d
            max_freq_T = self.multiresT - 1
            N_freqs_T = self.multiresT
            freq_bands_T = 2. ** torch.linspace(0., max_freq_T, steps=N_freqs_T).to(DEVICE)

            for freqT in freq_bands_T:
                for p_fn in self.embed_bases:
                    embed_fns_T.append(lambda x, p_fn=p_fn, freq=freqT: p_fn(x * torch.pi * freqT))
                    out_dim += d
            self.embed_fns_T = embed_fns_T
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def _embed_iso(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def _embed_aniso(self, inputs):
        embeded = self._embed_iso(inputs[...,:-1])
        embeded_Z = torch.cat([fn(inputs[...,-1:]) for fn in self.embed_fns_Z], -1)
        return torch.cat([embeded, embeded_Z], -1)

    def _embed_aniso1(self, inputs):
        embeded = self._embed_iso(inputs[...,:-2])
        embeded_Z = torch.cat([fn(inputs[...,-2:-1]) for fn in self.embed_fns_Z], -1)
        embeded_T = torch.cat([fn(inputs[..., -1:]) for fn in self.embed_fns_T], -1)
        return torch.cat([embeded, embeded_Z,embeded_T], -1)

