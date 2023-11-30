
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        record = self.hf_dataset[idx]
        sample = record['image'] if 'image' in record else record['img']
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        label = record['label'] if 'label' in record else record['fine_label']
        return sample, label