import h5py

# Open the HDF5 file
file_path = '/home/amisha/ibrl/release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5'

# Indices at which to split the demos
split_indices = [8, 8, 8, 8, 8]

# Open the original HDF5 file
with h5py.File(file_path, 'r') as f:
    # Create new HDF5 files for the split parts
    with h5py.File('/home/amisha/ibrl/release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/split_part1_11.hdf5', 'w') as f1, h5py.File('/home/amisha/ibrl/release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/split_part2_11.hdf5', 'w') as f2:
        # Iterate through the demos and split them
        for i, split_index in enumerate(split_indices):
            demo_name = f'demo_{i}'
            
            # Create groups for demo_i in both output files
            f1.create_group(f'data/{demo_name}')
            f2.create_group(f'data/{demo_name}')
            
            print(f"Actions Shape (demo_{i}): ", f[f'data/demo_{i}/actions'])
            # Split the datasets
            for dataset_name in ['actions', 'dones', 'rewards', 'states']:
                if dataset_name in f[f'data/{demo_name}']:
                    data = f[f'data/{demo_name}/{dataset_name}'][:]
                    f1.create_dataset(f'data/{demo_name}/{dataset_name}', data=data[:split_index+1])
                    f2.create_dataset(f'data/{demo_name}/{dataset_name}', data=data[split_index+1:])
            
            # Split the datasets within 'obs' group
            for obs_name in ['prop', 'corner2_image', 'state']:
                if obs_name in f[f'data/{demo_name}/obs']:
                    data = f[f'data/{demo_name}/obs/{obs_name}'][:]
                    f1.create_dataset(f'data/{demo_name}/obs/{obs_name}', data=data[:split_index+1])
                    f2.create_dataset(f'data/{demo_name}/obs/{obs_name}', data=data[split_index+1:])

print("Splitting completed and files are saved.")
