'''
Custom collate function for batching data samples.
It is required as batch of observation samples may have varying sequence lengths.
'''

import torch
import numpy as np
import torch.nn.functional as F


def _pad_batch(batch, pad_value=0):
    """
    Pads a batch of tensors to the maximum length in the batch.
    """
    N_max = max(t.shape[0] for t in batch)

    padded_batch = []
    for t in batch:
        pad_size = N_max - t.shape[0]
        # Pad on the N dimension: (pad_N_end, pad_N_start, pad_T_end, pad_T_start, pad_F_end, pad_F_start)
        # Since we only pad on N axis at the front (or end), we use: (0, 0, 0, 0, 0, pad_size)
        if len(t.shape) == 2:  # N, F
            padded = F.pad(t, (0, 0, 0, pad_size), value=pad_value)
        elif len(t.shape) == 3:  # N, T, F
            padded = F.pad(t, (0, 0, 0, 0, 0, pad_size), value=pad_value)  # Pad at end of N
        padded_batch.append(padded)

    return padded_batch


def custom_collate(batch):
    """
    Collates a batch of tuples where each item is (input_dict, target_array).
    """
    inputs, targets = zip(*batch)  # unzip the batch

    def collate_elem(elems):
        elem = elems[0]
        if isinstance(elem, torch.Tensor):
            return torch.stack(elems)
        elif isinstance(elem, np.ndarray):
            return torch.from_numpy(np.array(elems))
        elif isinstance(elem, (float, int, np.float32)):
            return torch.tensor(elems)
        elif isinstance(elem, str):
            return list(elems)
        # elif isinstance(elem, collections.abc.Mapping):
        #     return {k: collate_elem([d[k] for d in elems]) for k in elem}
        elif isinstance(elem, dict):
            return_d = {}
            for d in elems:
                for key in d.keys():
                    d_v = collate_elem(d[key])
                    if key in return_d:
                        return_d[key].append(d_v)
                    else:
                        return_d[key] = [d_v]

            # Pad the tensors in the dictionary to the maximum length
            return_d['historical_adjacent_obs'] = _pad_batch(return_d['historical_adjacent_obs'])
            return_d['hidden_ogm_cells'] = _pad_batch(return_d['hidden_ogm_cells'])
            return_d['seq_mask'] = _pad_batch(return_d['seq_mask'], pad_value=1)

            # convert list of tensors to a torch tensor
            for key in return_d.keys():
                if key != 'edge_index' and key != 'edge_weights':
                    return_d[key] = torch.stack(return_d[key])

            # create masks using the historical observations (Will be used to mask out irrelevant cells later)
            mask = (return_d['hidden_ogm_cells'] != 0).all(dim=-1)
            return_d['mask'] = mask

            return_d['vehicle_mask'] = (return_d['seq_mask'] != 0).all(dim=-1)

            return return_d

        else:
            raise TypeError(f"Unsupported type: {type(elem)}")

    # Collate inputs
    collated_inputs = collate_elem(inputs)

    # Collate targets
    target_collated = []
    for i, target in enumerate(targets):
        target_collated.append(collate_elem(target))
    collated_targets = torch.stack(_pad_batch(target_collated))  # Stack and pad targets

    return collated_inputs, collated_targets