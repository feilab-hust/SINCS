import torch
import torch.nn.functional as F
from glbSettings import *
import numpy as np



img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(DEVICE))





def get_block(pts, model, embedder, post=torch.relu, chunk=32*1024, model_kwargs={'embedder':{}, 'post':{}}):
    sh = pts.shape[:-1]
    pts = pts.reshape([-1,pts.shape[-1]])
    # break down into small chunk to avoid OOM
    outs = []
    for i in range(0, pts.shape[0], chunk):
        pts_chunk = pts[i:i+chunk]
        # eval
        out = model.eval(pts_chunk, embedder, post, model_kwargs)
        outs.append(out)
    outs = torch.cat(outs, dim=0)
    outs = outs.reshape([*sh,-1])
    return outs


def get_dblock(pts, t, model, embed_fn, embedtime_fn, post=torch.relu, chunk=32 * 1024,
               model_kwargs={'embed_fn': {}, 'embedtime_fn': {}, 'post': {}}):
    sh = pts.shape[:-1]
    pts = pts.reshape([-1, pts.shape[-1]])  # Reshape pts to match the shape of input_pts in DirectTemporalNeRF
    t = t.reshape([-1, t.shape[-1]])
    # break down into small chunks to avoid OOM
    outs = []
    for i in range(0, pts.shape[0], chunk):
        pts_chunk = pts[i:i + chunk]
        t_chunk = t[i:i + chunk]

        # 对点云进行嵌入处理
        # embedded_pts = embed_fn(pts_chunk)
        # embedded_t = embedtime_fn(t_chunk)

        # 使用model对象进行评估
        out = model.eval1(pts_chunk,t_chunk,post, model_kwargs)
        outs.append(out)
    outs = torch.cat(outs, dim=0)
    outs = outs.reshape([*sh, -1])
    return outs

def get_siren_block(pts, model, chunk=32*1024):
    sh = pts.shape[:-1]
    pts = pts.reshape([-1,pts.shape[-1]])
    # break down into small chunk to avoid OOM
    outs = []
    for i in range(0, pts.shape[0], chunk):
        pts_chunk = pts[i:i+chunk]
        # eval
        out = model.forward(pts_chunk)
        outs.append(out)
    outs = torch.cat(outs, dim=0)
    outs = outs.reshape([*sh,-1])
    return outs
def edge_loss(pred,target):
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    # padding=  [[0, 0], [1, 1], [1, 1], [0, 0]]
    kernel_x=torch.from_numpy(np.asarray(kernels[0],np.float32)).to(DEVICE)
    kernel_y = torch.from_numpy(np.asarray(kernels[1], np.float32)).to(DEVICE)
    # kernelsX_rep=torch.expand_copy()
    # kernelsY_rep =
    # # pred_paddi
    pred_edge_x = torch.nn.functional.conv2d(input=pred.permute(0, 3, 1, 2),
                                             weight=kernel_x.reshape(1, pred.shape[-1], 3, 3), padding=1)
    pred_edge_y = torch.nn.functional.conv2d(input=pred.permute(0, 3, 1, 2),
                                             weight=kernel_y.reshape(1, pred.shape[-1], 3, 3), padding=1)

    target_edge_x = torch.nn.functional.conv2d(input=target.permute(0, 3, 1, 2),
                                               weight=kernel_x.reshape(1, target.shape[-1], 3, 3), padding=1)
    target_edge_y = torch.nn.functional.conv2d(input=target.permute(0, 3, 1, 2),
                                               weight=kernel_y.reshape(1, target.shape[-1], 3, 3), padding=1)
    # pred_edge_x = torch.nn.functional.conv2d(input=pred[None, None, ..., 0], weight=kernel_x.reshape(-1, 1, 3, 3),
    #                                          padding=1)
    # pred_edge_y = torch.nn.functional.conv2d(input=pred[None, None, ..., 0], weight=kernel_y.reshape(-1, 1, 3, 3),
    #                                          padding=1)
    #
    # target_edge_x = torch.nn.functional.conv2d(input=target[None, None, ..., 0], weight=kernel_x.reshape(-1, 1, 3, 3),
    #                                            padding=1)
    # target_edge_y = torch.nn.functional.conv2d(input=target[None, None, ..., 0], weight=kernel_y.reshape(-1, 1, 3, 3),
    #                                            padding=1)

    return (img2mse(pred_edge_x, target_edge_x) + img2mse(pred_edge_y, target_edge_y)) / 2

def edge_loss1(pred, target):
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    kernel_x = torch.from_numpy(np.asarray(kernels[0], np.float32)).to(DEVICE)
    kernel_y = torch.from_numpy(np.asarray(kernels[1], np.float32)).to(DEVICE)

    pred_reshaped = pred.reshape(-1, 1, 1500, 1000)
    target_reshaped = target.reshape(-1, 1, 1500, 1000)

    pred_edge_x = torch.nn.functional.conv2d(input=pred_reshaped.permute(0, 1, 3, 2),
                                             weight=kernel_x.reshape(1, 1, 3, 3), padding=1)
    pred_edge_y = torch.nn.functional.conv2d(input=pred_reshaped.permute(0, 1, 3, 2),
                                             weight=kernel_y.reshape(1, 1, 3, 3), padding=1)

    target_edge_x = torch.nn.functional.conv2d(input=target_reshaped.permute(0, 1, 3, 2),
                                               weight=kernel_x.reshape(1, 1, 3, 3), padding=1)
    target_edge_y = torch.nn.functional.conv2d(input=target_reshaped.permute(0, 1, 3, 2),
                                               weight=kernel_y.reshape(1, 1, 3, 3), padding=1)

    return (img2mse(pred_edge_x, target_edge_x) + img2mse(pred_edge_y, target_edge_y)) / 2