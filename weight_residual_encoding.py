import os
import shutil
import torch
import tempfile


def process_and_save_params(source_folder, output_folder, restore_folder):
    index = 0

    target_folder = tempfile.mkdtemp()
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_folder):
        for name in files:
            if name == "The_Final.tar":
                source_file_path = os.path.join(root, name)
                target_file_name = "Group_" + "{:03d}.tar".format(index)
                target_file_path = os.path.join(target_folder, target_file_name)
                shutil.copy(source_file_path, target_file_path)
                index += 1


    shutil.copytree(target_folder, restore_folder)

    for i in range(0, index):
        current_file = os.path.join(target_folder, f'Group_{i:03d}.tar')
        previous_file = os.path.join(target_folder, f'Group_{i-1:03d}.tar')

        if i == 0:
            output_file = os.path.join(output_folder, f'Group_{i:03d}.tar')
            shutil.copy(current_file, output_file)
        else:
            output_file = os.path.join(output_folder, f'Group_{i:03d}-Group_{i-1:03d}.tar')

            params1 = torch.load(current_file)['network_fn_state_dict']
            params2 = torch.load(previous_file)['network_fn_state_dict']

            diff_params = {}
            for param in params1:
                if param in params2:
                    diff = params1[param] - params2[param]
                    diff_params[param] = diff

            new_params = {'network_fn_state_dict': diff_params}
            torch.save(new_params, output_file)

    shutil.rmtree(target_folder)


source_folder = "H:\\INR\\logs\\Compress_MT1"
output_folder = "H:\\INR\\logs\\Compress_MT1\\residual"
restore_folder = "H:\\INR\\logs\\Compress_MT1\\original"
process_and_save_params(source_folder, output_folder, restore_folder)