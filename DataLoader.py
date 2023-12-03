
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets

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
    
    
class CustomDataLoader:
    def __init__(self, args):
        self.dataset = None
        self.num_workers = args.workers
        self.image_size = args.image_size
        
    def get_dataset(self, dataset_name, batch_size=None, test_batch_size=None):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])
        
        if dataset_name == "imagenette":
            # use the imagenette dataset
            hf_dataset = load_dataset("frgfm/imagenette", '320px')
            hf_dataset = concatenate_datasets(hf_dataset.values())

        elif dataset_name == "cifar10":
            # use the cifar10 dataset
            hf_dataset = load_dataset("cifar10")
            hf_dataset = concatenate_datasets(hf_dataset.values())
            
            transform = transforms.Compose([
                *transform.transforms,
                transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                    std=[0.2673, 0.2564, 0.2761])
            ])
        elif "cifar100" in dataset_name:
            # use the cifar100 dataset
            hf_dataset = load_dataset("cifar100")
            hf_dataset = concatenate_datasets(hf_dataset.values())
        else:
            raise ValueError("Dataset not supported")
            
        transform = transforms.Compose([
            *transform.transforms,
            transforms.RandomHorizontalFlip()])
            
        dataset = CustomDataset(hf_dataset, transform=transform)
        test_size = 0.2
        test_volume = int(test_size * len(dataset))
        train_volume = len(dataset) - test_volume
        
        if batch_size is None:
            batch_size = 32 if dataset_name == "imagenette" else 64
        if test_batch_size is None:
            test_batch_size = batch_size

        train_dataset, test_dataset = random_split(dataset, [train_volume, test_volume])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=4
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_dataloader, test_dataloader, dataset