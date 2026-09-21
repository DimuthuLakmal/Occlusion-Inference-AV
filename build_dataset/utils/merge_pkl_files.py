import os
import os.path as osp
import pickle

# You can use this script to combine multiple observation pickle files in a directory into a single pickle file.
def combine_pickle_files(directory_path, output_file):
    all_content = {}
    file_names = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pkl'):
            file_path = osp.join(directory_path, file_name)
            file_names.append(file_path)
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                all_content = {**all_content, **content}

    with open(output_file, 'wb') as out:
        pickle.dump(all_content, out, protocol=pickle.HIGHEST_PROTOCOL)


# Example usage:
directory_path = '../../data/'
output_file = '../../data/data_v3.pkl'
combine_pickle_files(directory_path, output_file)
