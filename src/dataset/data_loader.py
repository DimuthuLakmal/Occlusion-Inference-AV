from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.collate_fn import custom_collate
from dataset.dataset import OGMDataset


class OGMDataLoader():
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.phase = phase

    def create_dataloader(self, phase=None):
        dataset = OGMDataset(self.cfg, self.phase)

        if self.phase == 'train':
            sampler = WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True,
            )
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.cfg['batch_size'],
                                    sampler=sampler,
                                    collate_fn=custom_collate,
                                    num_workers=self.cfg['num_workers'])
        else:
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.cfg['batch_size'],
                                    shuffle=False,
                                    collate_fn=custom_collate,
                                    num_workers=self.cfg['num_workers'])
        return dataloader
