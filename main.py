## Dependences
import torch
import os
from absl import flags, app

## User Defined Modules
from train_multi_4D import train_multi_4D
from train_multi_3D import train_multi_3D
from glbSettings import *




## Config Settings
# general
flags.DEFINE_enum('action', 'TRAIN_4D', ['TRAIN_3D','TRAIN_4D'], 'Action: TRAIN_3D or TRAIN_4D.')
flags.DEFINE_string('basedir', './logs/', 'Where to store ckpts and logs.')
flags.DEFINE_string('expname', 'Compress_output', help='Experiment name.')
flags.DEFINE_string('datadir', './data/dataset', 'Input data path.')
# embedder configs
flags.DEFINE_enum('embeddertype', 'PositionalEncoder', ['None','PositionalEncoder'], 'Encoder type.')
# PositionalEncoder
flags.DEFINE_integer('multires', 10, 'log2 of max freq for positional encoding.')
flags.DEFINE_integer('multiresZ', None, 'log2 of max freq for positional encoding on z axis for anisotropic encoding.')
flags.DEFINE_integer('multiresT', None, 'log2 of max freq for positional encoding on t axis for anisotropic encoding.')
flags.DEFINE_integer('i_embed', 0, 'set 0 for default positional encoding, -1 for none')
# model configs
flags.DEFINE_enum('modeltype', 'NeRF', ['NeRF','Siren'], 'Model type.')
flags.DEFINE_boolean('no_reload', False, 'Do not reload weights from saved ckpt.')
flags.DEFINE_string('weights_path', None, 'Weights to be loaded from.')
flags.DEFINE_integer('sigch', 1, '#channels of the signal to be predicted.')
# NeRF-like
flags.DEFINE_integer('netdepth', 6, '#Layers.')
flags.DEFINE_integer('netwidth',128, '#Channels per layer.')
flags.DEFINE_list('skips', [3], 'Skip connection layer indice.')
flags.DEFINE_integer('input_dim', 3, 'Latitude of model input.')
#siren
flags.DEFINE_integer('w0', 30, 'Frequency parameters of the sine function.')

# postprocessor configs
flags.DEFINE_enum('postprocessortype', 'relu', ['linear','relu','leak_relu'], 'Post processor type.')
# data options
flags.DEFINE_enum('datatype', 'multi_4D', ['multi_3D','multi_4D'], 'Dataset type:multi_4D.')
# training options
flags.DEFINE_integer('N_steps', 10000, 'Number of training steps.')
flags.DEFINE_float('lrate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('lrate_decay', 100, 'Exponential learning rate decay (in 1000 steps).')
flags.DEFINE_list('block_size',[10,10,10,10], 'Block Size trained one time. Should be an odd number.')
flags.DEFINE_integer('size_2D',100000, 'Vector Size trained one time. Should be an odd number.')
# rendering options
flags.DEFINE_integer('chunk', 64*1024, 'Number of pts sent through network in parallel, decrease if running out of memory.')
flags.DEFINE_list('render_size', None, 'Size of the extracted volume, should be in format "W,H,D". "None" for extract autometically from training data.')
# logging
flags.DEFINE_integer('i_weights', 10000, 'Weight ckpt saving interval.')
flags.DEFINE_integer('i_pre_weights', 10000, 'Weight ckpt saving interval.')
flags.DEFINE_integer('i_print', 1000, 'Printout and logging interval.')
flags.DEFINE_integer('i_pre_print', 1000, 'Printout and logging interval.')
flags.DEFINE_integer('step_size', 3, 'Group numbers.')
flags.DEFINE_float('threshold', 0.1, 'Threshold for generating saliency maps')

FLAGS=flags.FLAGS

def main(argv):
    del argv
    if FLAGS.action.lower() == 'train_3d':
        train_multi_3D(FLAGS)
    elif FLAGS.action.lower() == 'train_4d':
        train_multi_4D(FLAGS)

    else:
        print("Unrecognized action. Do nothing.")



if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # np.random.seed(0)
    
    if DEVICE.type == 'cuda':
        print(f"Run on CUDA:{GPU_IDX}")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("Run on CPU")
    torch.set_default_tensor_type('torch.FloatTensor')

    app.run(main)