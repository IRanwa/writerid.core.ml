import os
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WriterDataset(Dataset):
    def __init__(self, dataset_path: str, writers_list: List[str], 
                 transform: Optional[transforms.Compose] = None, 
                 image_size: int = 224,
                 mode: str = 'images',
                 dataset_path_override: Optional[str] = None):
        
        self.dataset_path = dataset_path_override if dataset_path_override else dataset_path
        self.writers = sorted(writers_list)
        self.transform = transform
        self.image_size = image_size
        self.mode = mode

        self.item_paths: List[str] = []
        self.item_labels: List[int] = []
        self.writer_to_label: Dict[str, int] = {writer_id: i for i, writer_id in enumerate(self.writers)}
        self.label_to_writer: Dict[int, str] = {i: writer_id for writer_id, i in self.writer_to_label.items()}
        
        file_extension = '.pt' if self.mode == 'embeddings' else ''

        for writer_id in self.writers:
            writer_dir = os.path.join(self.dataset_path, writer_id)
            try:
                for item_file in os.listdir(writer_dir):
                    if file_extension and not item_file.endswith(file_extension):
                        continue
                    item_path = os.path.join(writer_dir, item_file)
                    self.item_paths.append(item_path)
                    self.item_labels.append(self.writer_to_label[writer_id])
            except FileNotFoundError:
                print(f"Warning: Writer directory not found {writer_dir} for dataset {self.dataset_path}")
            except Exception as e:
                print(f"Warning: Could not read files for writer {writer_id} in {writer_dir}: {e}")
        
        if not self.item_paths:
            raise ValueError(f"No valid items (mode: {self.mode}) found for the provided writers in {self.dataset_path}")

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item_path = self.item_paths[idx]
        label = self.item_labels[idx]
        
        try:
            if self.mode == 'images':
                item = Image.open(item_path).convert('L')
                if self.transform:
                    item = self.transform(item)
            elif self.mode == 'embeddings':
                item = torch.load(item_path)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        except Exception as e:
            print(f"Error loading/transforming {item_path} in mode '{self.mode}': {e}. Returning placeholder.")
            if self.mode == 'images':
                num_channels = 1
                item = torch.zeros((num_channels, self.image_size, self.image_size))
            else:
                item = torch.zeros((1000,))
                
        return item, label

    def get_labels(self) -> List[int]:
        return self.item_labels

class DatasetManager:
    def __init__(self, dataset_path: str, train_ratio: float, seed: Optional[int] = None):
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

    def _load_writers_from_path(self) -> List[str]:
        writers = []
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        for writer_folder_name in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, writer_folder_name)):
                writers.append(writer_folder_name)
        if not writers:
            raise ValueError(f"No writer subdirectories found in {self.dataset_path}")
        return writers

    def get_train_test_writers(self) -> Tuple[List[str], List[str]]:
        print(f"Loading writers from: {self.dataset_path}")
        writers = self._load_writers_from_path()
        print(f"Found {len(writers)} writers in total.")
        if self.seed is not None:
             random.seed(self.seed)
        random.shuffle(writers)
        n_total = len(writers)
        n_train = int(n_total * self.train_ratio)
        train_writers = writers[:n_train]
        test_writers = writers[n_train:]
        print(f"Split: {len(train_writers)} train, {len(test_writers)} test writers.")
        return train_writers, test_writers

    def get_writer_dataset(self, writers_list: List[str], transform: Optional[transforms.Compose], 
                             image_size: int, mode: str = 'images', 
                             dataset_path_override: Optional[str] = None) -> WriterDataset:
        return WriterDataset(self.dataset_path, writers_list, transform, image_size, mode, dataset_path_override)