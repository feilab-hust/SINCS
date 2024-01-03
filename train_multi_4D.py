import torch
import numpy as np
from tqdm import tqdm, trange
import tifffile
import os
from core.load import load_data
from core.models.Model import create_model
from core.utils.misc import *
from core.utils.Coord import Coord
from glbSettings import *
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
import time




def train_multi_4D(Flags):
    images, maxval,saliency_map= load_data(path=Flags.datadir, datatype=Flags.datatype, step_size=Flags.step_size,threshold=Flags.threshold,normalize=True, ret_max_val=True)


    images1=images[0]
    img_raw = images[0]
    img_raw_array = np.array(img_raw)
    images1 = np.expand_dims(images1, axis=-1)
    images1 = np.transpose(images1, (1, 2, 3, 0, 4))
    D, H, W,T,C = images1.shape

    brXY = eval(Flags.block_size[0] )// 2
    brZ = eval(Flags.block_size[2]) // 2
    brT = eval(Flags.block_size[3]) // 2



    S = np.array([W, H, D], dtype=np.float32)
    maxHW = max(H, W)
    sc = np.full(3, 2 / (maxHW - 1), dtype=np.float32)
    dc = -((S // 2) * 2) / (maxHW - 1)
    dc[1] *= -1  # flip Y
    sc[1] *= -1

    dc[2] *= -1  # flip Z
    sc[2] *= -1
    Coord.set_idx2glb_scale(sc)
    Coord.set_idx2glb_trans(dc)

    ## Create grid indice
    ## Save model for each image
    for i, img in enumerate(images):
        img_name = f"image_{i:03d}"
        img_dir = os.path.join(Flags.basedir, Flags.expname, img_name)
        os.makedirs(img_dir, exist_ok=True)


        Flags.append_flags_into_file(os.path.join(Flags.basedir, Flags.expname,img_name, 'config.cfg'))


        lrate=  Flags.lrate

        if Flags.weights_path is not None and Flags.weights_path != 'None':
            ckpts = [Flags.weights_path]
        else:
            ckpts = [os.path.join(Flags.basedir, Flags.expname,img_name, f) for f in
                     sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname,img_name))) if 'cp_' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not Flags.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
        else:
            ckpt_path = None
            print("Creating new model...")

        model, embedder, post_processor, optimizer, start = create_model(Flags, (D, H, W), Flags.modeltype,
                                                                         Flags.embeddertype, Flags.postprocessortype,
                                                                         lr=Flags.lrate, weights_path=ckpt_path)

        # #################################


        images[i]= np.expand_dims(images[i], axis=-1)
        img1=images[i]
        img1 = np.transpose(img1, (1, 2, 3, 0, 4))
        saliency_map = np.array(saliency_map)
        saliency_map1=saliency_map[i]
        maxval1=maxval[i]
        indices = np.where(saliency_map1 == 1)
        points = np.column_stack((indices[1], indices[2], indices[3], indices[0]))
        num_points = len(points)



        saliency_map_points_set = torch.tensor(saliency_map1[indices]).float().reshape(num_points, 1)
        sum = torch.sum(saliency_map_points_set)
        saliency_map_normalized = saliency_map_points_set / sum

        points_set = torch.tensor(saliency_map1[indices]).float().reshape(num_points, 1)
        indices = np.transpose(indices, (1, 0))
        original_points_set = torch.tensor(img1)[indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 0]]
        original_points_set = original_points_set.float().reshape(num_points, 1)
        indices = torch.Tensor(indices).to(DEVICE)
        pts_glb_all1 = Coord.idx2glb(indices[:, 1:])

        indices_0 = torch.linspace(-1, 1, num_points)
        indices_0 = torch.Tensor(indices_0).to(DEVICE)


        pts_glb_all2 = torch.cat((indices_0.unsqueeze(1), pts_glb_all1), dim=1)

        original_points_set = torch.Tensor(original_points_set).to(DEVICE)

        #################################
        ##          CORE LOOP          ##
        #################################
        ## run
        print("Start training...")
        start += 1




        if i > 0:
            N_steps = int(Flags.N_steps/4*3 + 1)
        else:
            N_steps = Flags.N_steps + 1
        for step in trange(start, N_steps):

            # #

            # #################################
            saliency_map_1d = saliency_map_normalized.view(-1)
            saliency_map_1d = saliency_map_1d.to(DEVICE)
            pts_glb_random_indices = torch.randint(high=saliency_map_1d.size(0), size=(Flags.size_2D,))

            pts_glb = pts_glb_all2[pts_glb_random_indices, :]
            block = get_block(pts_glb, model, embedder, post_processor, Flags.chunk)
            print('mean: %.6f' % torch.mean(block.data).cpu().numpy())
            target = original_points_set[pts_glb_random_indices, :]
            # #################################

            # updating weights
            optimizer.zero_grad()

            img_loss = img2mse(block, target)



            loss = img_loss   # add any other type of loss here
            loss.backward()
            optimizer.step()

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = Flags.lrate_decay * 1000
            new_lrate = lrate * (decay_rate ** ((step - 1) / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate


            ################################
            ## logging
            if step % Flags.i_weights == 0:
                path = os.path.join(Flags.basedir, Flags.expname, img_name,'cp_{:06d}.tar'.format(step))
                save_dict = {
                    'global_step': step - 1,
                    'network_fn_state_dict': model.get_state(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(save_dict, path)
                print('Saved checkpoints at', path)

            if step % Flags.i_print == 0:
                with torch.no_grad():
                    # TODO local block validation

                    #################################
                    voxel = get_block(pts_glb_all2, model, embedder, post_processor, chunk=Flags.chunk)
                    img = torch.Tensor(img_raw_array).to(DEVICE)
                    output_recon = torch.zeros_like(img)
                    indices = indices.long()
                    voxel = voxel.squeeze(-1)

                    output_recon[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = voxel
                    loss_avg = img2mse(block, target)
                    psnr_avg = mse2psnr(loss_avg)
                    #################################
                    #
                    # loss_avg = img2mse(output_recon,images)
                    tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {loss_avg.item()} PSNR: {psnr_avg.item()}"
                    tqdm.write(tqdm_txt)
                    # write into file
                    path = os.path.join(Flags.basedir, Flags.expname,img_name, 'logs.txt')
                    with open(path, 'a') as fp:
                        fp.write(tqdm_txt + '\n')
                    if step % (N_steps-1) == 0:
                        last_path = os.path.join(Flags.basedir, Flags.expname, img_name, 'The_Final.tar'.format(step))
                        save_dict = {
                            # 'global_step': step - 1,
                            'network_fn_state_dict': model.get_state(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                        }
                        torch.save(save_dict, last_path)
                        print('Saved checkpoints at', last_path
                              )
                        decompression_dir = os.path.join(Flags.basedir, Flags.expname, f"decompression")
                        os.makedirs(decompression_dir, exist_ok=True)
                        stride=Flags.step_size
                        for j in range(output_recon.shape[0]):
                            current_slice = output_recon[j]* maxval1.item()
                            current_slice = current_slice - Flags.threshold
                            current_slice = torch.nn.functional.relu(current_slice)

                            save_index = j * stride + i
                            tifffile.imwrite(
                                os.path.join(Flags.basedir, Flags.expname, f"decompression", f'Rendering_{save_index:03d}.tif'),
                                np.squeeze(current_slice.cpu().numpy().astype(np.uint16))
                            )
            if step % (N_steps-1) == 0 and (i + 1) % 3 != 0:
                # Update 'global_step' to 1
                save_dict = {
                    'global_step':  1,
                    'network_fn_state_dict': model.get_state(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }

                model_state_dict = save_dict['network_fn_state_dict']


                model_state_dict['pts_linear.0.weight'] = torch.zeros_like(model_state_dict['pts_linear.0.weight'])
                model_state_dict['pts_linear.0.bias'] = torch.zeros_like(model_state_dict['pts_linear.0.bias'])
                model_state_dict['output_linear.weight'] = torch.zeros_like(model_state_dict['output_linear.weight'])
                model_state_dict['output_linear.bias'] = torch.zeros_like(model_state_dict['output_linear.bias'])

                target_layer_index = Flags.netdepth - 1
                weight_key = f'pts_linear.{target_layer_index}.weight'
                bias_key = f'pts_linear.{target_layer_index}.bias'
                model_state_dict[weight_key] = torch.zeros_like(model_state_dict[weight_key])
                model_state_dict[bias_key] = torch.zeros_like(model_state_dict[bias_key])

                save_dict['network_fn_state_dict'] = model_state_dict

                # Update path to the next image folder
                next_img_name = 'image_{:03d}'.format(i + 1)
                img_dir1 = os.path.join(Flags.basedir, Flags.expname, next_img_name)
                os.makedirs(img_dir1, exist_ok=True)
                path = os.path.join(Flags.basedir, Flags.expname, next_img_name,'cp_000001.tar')

                torch.save(save_dict, path)
                print('Saved checkpoints at', path)


            del loss, img_loss, target
        torch.cuda.empty_cache()


