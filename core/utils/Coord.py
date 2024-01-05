import torch
import numpy as np



class Coord:
    idx2glb_scale = None    # (w,h,d)->(x,y,z)
    idx2glb_trans = None    # (w,h,d)->(x,y,z)

    @ staticmethod
    def idx2glb(idx):
        # (d,h,w)->(w,h,d)
        if type(idx) is torch.Tensor:
            is_tensor, device = True, idx.device
            idx = np.array(idx.to('cpu'), dtype=np.float32)[...,::-1].copy()
        elif type(idx) is np.ndarray:
            is_tensor = False
            idx = idx[...,::-1]
        else:
            raise TypeError("Support torch.Tensor and numpy.ndarray only.")
        glb = idx*Coord.idx2glb_scale + Coord.idx2glb_trans # (w,h,d)->(x,y,z)
        return torch.from_numpy(glb).to(device) if is_tensor else glb

    @ staticmethod
    def glb2idx(glb):
        if type(glb) is torch.Tensor:
            is_tensor, device = True, glb.device
            glb = np.array(glb.to('cpu'), dtype=np.float32)
        elif type(glb) is np.ndarray:
            is_tensor = False
            glb = glb
        else:
            raise TypeError("Support torch.Tensor and numpy.ndarray only.")
        idx = ((glb - Coord.idx2glb_trans) / Coord.idx2glb_scale).round().astype(np.int64)  # (x,y,z)->(w,h,d)
        idx = idx[...,::-1] # (w,h,d)->(d,h,w)
        return torch.from_numpy(idx.copy()).to(device) if is_tensor else idx
        
    @ staticmethod
    def get_idx2glb_scale():
        return Coord.idx2glb_scale

    @ staticmethod
    def get_idx2glb_trans():
        return Coord.idx2glb_trans

    @ staticmethod
    def get_glb2idx_scale():
        return 1. / Coord.idx2glb_scale

    @ staticmethod
    def get_glb2idx_trans():
        return -Coord.idx2glb_trans / Coord.idx2glb_scale

    @ staticmethod
    def set_idx2glb_scale(idx2glb_scale):
        Coord.idx2glb_scale = idx2glb_scale

    @ staticmethod
    def set_idx2glb_trans(idx2glb_trans):
        Coord.idx2glb_trans = idx2glb_trans


