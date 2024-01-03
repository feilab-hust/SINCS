import os

import torch


def restore_and_save_params(input_folder, output_folder):
    all_files = os.listdir(input_folder)


    tar_files = [file for file in all_files if file.endswith('.tar')]

    index=len(tar_files)


    os.makedirs(output_folder, exist_ok=True)


    for i in range(0, index):


        if i == 0:
            raw_file = os.path.join(input_folder, f'Group_{i:03d}.tar')
            output_file = os.path.join(output_folder, f'Group_{i:03d}.tar')
            diff_params = torch.load(raw_file)['network_fn_state_dict']
            torch.save(diff_params, output_file)
        else:

            input_file = os.path.join(input_folder, f'Group_{i:03d}-Group_{i-1:03d}.tar')
            diff_params1 = torch.load(input_file)['network_fn_state_dict']
            if i ==1:
                previous_file = os.path.join(input_folder, f'Group_{i-1:03d}.tar')
            else:
                previous_file = os.path.join(output_folder, f'Group_{i-1:03d}.tar')
            params1 = torch.load(previous_file)['network_fn_state_dict']
            output_file1 = os.path.join(output_folder, f'Group_{i:03d}.tar')


            restored_params = {}
            for param in params1:
                if param in diff_params1:
                    restored = params1[param] + diff_params1[param]
                    restored_params[param] = restored

            restored_params_dict = {'network_fn_state_dict': restored_params}
            torch.save(restored_params_dict, output_file1)


input_folder = "H:\\INR\\logs\\Compress_multi4D_cart\\residual"
output_folder = "H:\\INR\\logs\\Compress_multi4D_cart\\residual\\decompressed"
restore_and_save_params(input_folder, output_folder)
