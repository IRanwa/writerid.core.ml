import random
from typing import Dict, List, Iterator, Tuple, Union
import torch
from torch.utils.data import Sampler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

class EpisodicTaskSampler(Sampler):
    def __init__(
        self,
        dataset: TorchDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        
        all_labels = dataset.get_labels()
        if not all_labels:
            raise ValueError("Dataset returned an empty list of labels.")

        items_per_label_all: Dict[int, List[int]] = {}
        for item_idx, label in enumerate(all_labels):
            if label not in items_per_label_all:
                items_per_label_all[label] = []
            items_per_label_all[label].append(item_idx)

        self.eligible_items_per_label: Dict[int, List[int]] = {}
        required_samples = self.n_shot + self.n_query
        skipped_labels_count = 0
        original_label_count = len(items_per_label_all)

        for label, items in items_per_label_all.items():
            if len(items) >= required_samples:
                self.eligible_items_per_label[label] = items
            else:
                writer_id_str = label
                if hasattr(dataset, 'label_to_writer') and isinstance(dataset.label_to_writer, dict) and label in dataset.label_to_writer:
                    writer_id_str = dataset.label_to_writer[label]
                
                print(f"Warning: Writer ID (label: {writer_id_str}) has {len(items)} samples, but {required_samples} are required (n_shot={self.n_shot} + n_query={self.n_query}). Skipping this writer for episodic sampling.")
                skipped_labels_count += 1
        
        if skipped_labels_count > 0:
            print(f"Info: Skipped {skipped_labels_count} out of {original_label_count} writers due to insufficient samples for the current N-shot/N-query configuration.")

        self._check_dataset_has_enough_eligible_labels()

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        available_eligible_labels = list(self.eligible_items_per_label.keys())
        if len(available_eligible_labels) < self.n_way:
            raise ValueError(
                f"Cannot form episodes. Dataset has only {len(available_eligible_labels)} eligible labels "
                f"(after filtering those with < {self.n_shot + self.n_query} samples), "
                f"but n_way={self.n_way} is requested."
            )

        for _ in range(self.n_tasks):
            sampled_labels = random.sample(available_eligible_labels, self.n_way)
            batch_indices = []
            
            for label in sampled_labels:
                items_for_this_label = self.eligible_items_per_label[label]
                sampled_indices = random.sample(items_for_this_label, self.n_shot + self.n_query)
                batch_indices.extend(sampled_indices)
                
            if len(batch_indices) == self.n_way * (self.n_shot + self.n_query):
                yield batch_indices

    def _check_dataset_has_enough_eligible_labels(self):
        num_eligible_labels = len(self.eligible_items_per_label)
        if self.n_way > num_eligible_labels:
            raise ValueError(
                f"Cannot form N-way episodes. Labels in dataset eligible for sampling ({num_eligible_labels}) "
                f"(after filtering those with < {self.n_shot + self.n_query} samples) "
                f"are fewer than n_way ({self.n_way}).")

    @staticmethod
    def _cast_input_data_to_tensor_int_tuple(
        input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> List[Tuple[Tensor, int]]:
        output_data = []
        for image, label in input_data:
            if not isinstance(image, Tensor): 
                raise TypeError(f"Bad image type: {type(image)}.")
            processed_label = label
            if not isinstance(label, int):
                if not isinstance(label, Tensor): 
                    raise TypeError(f"Bad label type: {type(label)}.")
                if label.dtype not in {torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64}: 
                    raise TypeError(f"Bad label tensor dtype: {label.dtype}.")
                if label.ndim != 0: 
                    raise ValueError(f"Bad label tensor shape: {label.shape}.")
                processed_label = int(label.item())
            output_data.append((image, processed_label))
        return output_data

    def episodic_collate_fn(self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]: 
        input_data_with_int_labels = self._cast_input_data_to_tensor_int_tuple(input_data)
        true_class_ids = sorted(list({x[1] for x in input_data_with_int_labels}))
        current_n_way = len(true_class_ids)
        if current_n_way != self.n_way:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), []
        all_images_list = [x[0].unsqueeze(0) for x in input_data_with_int_labels]
        if not all_images_list : 
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), []
        all_images = torch.cat(all_images_list)
        
        expected_total_samples = current_n_way * (self.n_shot + self.n_query)
        if all_images.shape[0] != expected_total_samples:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), []
        try:
            all_images = all_images.reshape((current_n_way, self.n_shot + self.n_query, *all_images.shape[1:]))
        except:
             return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), []
        
        label_map = {true_id: i for i, true_id in enumerate(true_class_ids)}
        all_relative_labels = torch.tensor(
            [label_map[x[1]] for x in input_data_with_int_labels]
        ).reshape((current_n_way, self.n_shot + self.n_query))
        
        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_relative_labels[:, : self.n_shot].flatten()
        query_labels = all_relative_labels[:, self.n_shot :].flatten()
        
        return (support_images, support_labels, query_images, query_labels, true_class_ids,)