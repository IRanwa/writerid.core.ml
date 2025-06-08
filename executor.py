import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import os
import random
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple
from PIL import Image

from data_manager import DatasetManager
from image_processor import ImagePreprocessor
from sampler import EpisodicTaskSampler
from network_architectures import BackboneNetworkHandler
from models import PrototypicalNetworkModel
from training_utils import fit_episode, evaluate_model, sliding_average

class TaskExecutor:
    def __init__(self, run_config: Dict, n_way: int, n_shot: int, n_query: int,
                 n_training_episodes: int, n_evaluation_tasks: int, learning_rate: float,
                 backbone_name: str = "googlenet", pretrained_backbone: bool = True,
                 seed: Optional[int] = 42,
                 evaluation_interval: int = 100,
                 early_stopping_patience: int = 10,
                 model_save_path: Optional[str] = None,
                 embeddings_save_path: Optional[str] = None):

        self.config = run_config.copy()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_training_episodes = n_training_episodes
        self.n_evaluation_tasks = n_evaluation_tasks
        self.learning_rate = learning_rate
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone
        self.seed = seed
        self.evaluation_interval = evaluation_interval
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.config['device'] = str(self.device)
        self.model_save_path = model_save_path
        self.embeddings_save_path = embeddings_save_path

        self.dataset_manager: Optional[DatasetManager] = None
        self.preprocessor: Optional[ImagePreprocessor] = None
        self.backbone_handler: Optional[BackboneNetworkHandler] = None
        self.proto_model: Optional[PrototypicalNetworkModel] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.val_loader_for_early_stopping: Optional[DataLoader] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self._set_seeds_if_provided()
        
    def _set_seeds_if_provided(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def setup_pipeline(self):
        specific_dataset_path = self.config['dataset_path']
        self.dataset_manager = DatasetManager(
            dataset_path=specific_dataset_path,
            train_ratio=self.config['train_ratio'],
            seed=self.seed
        )
        train_writers, test_writers = self.dataset_manager.get_train_test_writers()
        if not train_writers or not test_writers:
            raise RuntimeError(f"Failed to get train/test writers from {specific_dataset_path}")

        self.backbone_handler = BackboneNetworkHandler(
            name=self.backbone_name,
            pretrained=self.pretrained_backbone
        )
        actual_backbone_module = self.backbone_handler.get_model().to(self.device)

        self.preprocessor = ImagePreprocessor(
            image_size=self.config['image_size']
        )
        train_transform = self.preprocessor.get_train_transform()
        test_transform = self.preprocessor.get_test_transform()

        try:
            train_set = self.dataset_manager.get_writer_dataset(train_writers, train_transform, self.config['image_size'])
            test_set = self.dataset_manager.get_writer_dataset(test_writers, test_transform, self.config['image_size'])
        except ValueError as e:
            raise RuntimeError(f"Error creating WriterDataset instances: {e}") from e
        
        print(f"Train set: {len(train_set)} images for {len(train_writers)} writers.")
        print(f"Test set: {len(test_set)} images for {len(test_writers)} writers.")

        try:
            train_sampler = EpisodicTaskSampler(train_set, self.n_way, self.n_shot, self.n_query, self.n_training_episodes)
            val_tasks = max(1, self.n_evaluation_tasks // 10 if self.n_evaluation_tasks >= 10 else self.n_evaluation_tasks)
            if val_tasks == 0 and self.n_evaluation_tasks > 0: 
                val_tasks = 1
            self.val_sampler_for_early_stopping = EpisodicTaskSampler(test_set, self.n_way, self.n_shot, self.n_query, val_tasks)
        except ValueError as e:
            raise RuntimeError(f"Error creating TaskSampler: {e}.") from e

        self.train_loader = DataLoader(
            train_set, batch_sampler=train_sampler, num_workers=self.config['num_workers'],
            pin_memory=(self.device.type == 'cuda'), collate_fn=train_sampler.episodic_collate_fn
        )
        self.val_loader_for_early_stopping = DataLoader(
            test_set, batch_sampler=self.val_sampler_for_early_stopping, num_workers=self.config['num_workers'],
            pin_memory=(self.device.type == 'cuda'), collate_fn=self.val_sampler_for_early_stopping.episodic_collate_fn
        )
        
        self.proto_model = PrototypicalNetworkModel(actual_backbone_module).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.proto_model.parameters(), lr=self.learning_rate)

    def train(self) -> Tuple[List[float], int, int, float, Optional[Dict]]:
        if not all([self.proto_model, self.train_loader, self.optimizer, self.criterion, self.val_loader_for_early_stopping]):
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")

        print(f"\n--- Executor: Starting Training (Max: {self.n_training_episodes} episodes) ---")
        print(f"Early stopping: Patience={self.early_stopping_patience} checks (1 check per {self.evaluation_interval} episodes).")

        all_loss: List[float] = []
        log_update_frequency = 10
        best_val_accuracy = -1.0
        patience_counter = 0
        best_model_state_dict = None
        actual_episodes_run = 0
        optimal_episode_for_val_acc = 0

        self.proto_model.train()
        train_iter = iter(self.train_loader)

        with tqdm(range(self.n_training_episodes), desc=f"Training {os.path.basename(self.config['dataset_path'])}") as pbar:
            for episode_index in pbar:
                actual_episodes_run = episode_index + 1
                try:
                    batch_data = next(train_iter)
                except StopIteration:
                    print(f"Train loader exhausted at episode {actual_episodes_run}. Training stopped.")
                    break
                
                if not batch_data or len(batch_data) != 5: continue
                support_images, support_labels, query_images, query_labels, _ = batch_data
                if support_images.numel() == 0 or query_images.numel() == 0: continue

                loss_value = fit_episode(self.proto_model, self.optimizer, self.criterion,
                                         support_images, support_labels, query_images, query_labels, self.device)

                if loss_value is not None: all_loss.append(loss_value)

                current_postfix = {}
                if all_loss: current_postfix['loss'] = f"{sliding_average(all_loss, log_update_frequency):.4f}"
                pbar.set_postfix(current_postfix)

                if (episode_index + 1) % self.evaluation_interval == 0 and (episode_index + 1) > 0:
                    val_metrics_dict = evaluate_model(self.proto_model, self.val_loader_for_early_stopping, self.device, self.n_way)
                    val_accuracy = val_metrics_dict.get("accuracy", 0.0)
                    self.proto_model.train()

                    pbar_desc = f"Ep {actual_episodes_run} ValAcc:{val_accuracy:.2f}%"
                    pbar.set_description_str(pbar_desc)

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        best_model_state_dict = copy.deepcopy(self.proto_model.state_dict())
                        optimal_episode_for_val_acc = actual_episodes_run
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered at episode {actual_episodes_run}.")
                        break
        
        if best_model_state_dict is not None:
            self.proto_model.load_state_dict(best_model_state_dict)
        
        print("--- Executor: Training Finished ---")
        return all_loss, actual_episodes_run, optimal_episode_for_val_acc, best_val_accuracy, best_model_state_dict

    def pre_compute_and_save_test_embeddings(self, model_state_dict: Dict) -> str:
        """Pre-computes and saves embeddings for the entire test set."""
        print("\n--- Executor: Pre-computing test set embeddings ---")
        if not self.embeddings_save_path:
            raise ValueError("embeddings_save_path must be provided to save embeddings.")

        self.proto_model.load_state_dict(model_state_dict)
        self.proto_model.to(self.device)
        self.proto_model.eval()

        _, test_writers = self.dataset_manager.get_train_test_writers()
        test_transform = self.preprocessor.get_test_transform()

        class SimpleImageDataset(Dataset):
            def __init__(self, root_path, writers, transform):
                self.paths = []
                for writer_id in writers:
                    writer_dir = os.path.join(root_path, writer_id)
                    for img_file in os.listdir(writer_dir):
                        self.paths.append(os.path.join(writer_dir, img_file))
                self.transform = transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img_path = self.paths[idx]
                image = Image.open(img_path).convert('L')
                return self.transform(image), img_path

        image_dataset = SimpleImageDataset(self.config['dataset_path'], test_writers, test_transform)
        image_loader = DataLoader(image_dataset, batch_size=32, num_workers=self.config['num_workers'])

        with torch.no_grad():
            for images, paths in tqdm(image_loader, desc="Generating Embeddings"):
                images = images.to(self.device)
                embeddings = self.proto_model.backbone(images)
                
                for i, full_path in enumerate(paths):
                    rel_path = os.path.relpath(full_path, self.config['dataset_path'])
                    embedding_save_path = os.path.join(self.embeddings_save_path, rel_path)
                    embedding_save_path = os.path.splitext(embedding_save_path)[0] + ".pt"
                    
                    os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)
                    torch.save(embeddings[i].cpu(), embedding_save_path)
        
        print(f"All test embeddings saved to: {self.embeddings_save_path}")
        return self.embeddings_save_path

    def evaluate_with_embeddings(self, embeddings_path: str) -> Dict:
        """Evaluates the model using pre-computed embeddings."""
        print("\n--- Executor: Evaluating with pre-computed embeddings ---")
        _, test_writers = self.dataset_manager.get_train_test_writers()

        try:
            embedding_dataset = self.dataset_manager.get_writer_dataset(
                writers_list=test_writers,
                transform=None,
                image_size=self.config['image_size'],
                mode='embeddings',
                dataset_path_override=embeddings_path 
            )
            
            test_sampler = EpisodicTaskSampler(embedding_dataset, self.n_way, self.n_shot, self.n_query, self.n_evaluation_tasks)
            test_loader = DataLoader(
                embedding_dataset, batch_sampler=test_sampler, num_workers=self.config['num_workers'],
                pin_memory=(self.device.type == 'cuda'), collate_fn=test_sampler.episodic_collate_fn
            )
        except (ValueError, FileNotFoundError) as e:
             raise RuntimeError(f"Error creating dataset/sampler for embeddings: {e}") from e

        metrics_dict = evaluate_model(test_loader, self.device, self.n_way)
        return metrics_dict

    def run_single_experiment(self) -> Dict:
        start_time = time.perf_counter()
        evaluation_metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": None}
        actual_episodes_run = 0
        optimal_val_episode = 0
        best_val_acc_from_training = 0.0
        error_message = None
        best_model_state = None

        try:
            self.setup_pipeline()
            _, actual_episodes_run, optimal_val_episode, best_val_acc_from_training, best_model_state = self.train()
            
            if best_model_state is None and actual_episodes_run > 0:
               print("No best model from validation was found. Using model from the final episode.")
               best_model_state = self.proto_model.state_dict()

            if best_model_state and self.model_save_path:
                os.makedirs(self.model_save_path, exist_ok=True)
                ep_for_filename = optimal_val_episode if optimal_val_episode > 0 else actual_episodes_run
                acc_for_filename = best_val_acc_from_training if best_val_acc_from_training > 0 else 0.0
                filename = f"{self.backbone_name}_ep{ep_for_filename}_acc{acc_for_filename:.2f}.pth"
                full_path = os.path.join(self.model_save_path, filename)
                torch.save(best_model_state, full_path)
                print(f"Best model state permanently saved to: {full_path}")

            if best_model_state:
                print("\n--- Executor: Evaluating best model on test set (image mode) ---")
                self.proto_model.load_state_dict(best_model_state)
                self.proto_model.eval()

                _, test_writers = self.dataset_manager.get_train_test_writers()
                test_transform = self.preprocessor.get_test_transform()
                
                final_test_set = self.dataset_manager.get_writer_dataset(
                    writers_list=test_writers, 
                    transform=test_transform, 
                    image_size=self.config['image_size'],
                    mode='images'
                )

                if len(final_test_set) == 0:
                    print("Warning: Final test set is empty. Skipping final evaluation.")
                    evaluation_metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": None}
                else:
                    try:
                        final_test_sampler = EpisodicTaskSampler(
                            final_test_set, self.n_way, self.n_shot, self.n_query, 
                            self.n_evaluation_tasks
                        )
                        final_test_loader = DataLoader(
                            final_test_set, batch_sampler=final_test_sampler, 
                            num_workers=self.config['num_workers'],
                            pin_memory=(self.device.type == 'cuda'), 
                            collate_fn=final_test_sampler.episodic_collate_fn
                        )
                        evaluation_metrics = evaluate_model(self.proto_model, final_test_loader, self.device, self.n_way)
                    except ValueError as e:
                        print(f"Error creating sampler/loader for final test evaluation: {e}. Skipping final evaluation.")
                        error_message = error_message + f" | Final Test Eval Error: {str(e)}" if error_message else f"Final Test Eval Error: {str(e)}"
                        evaluation_metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": None}

            else:
                print("Skipping final evaluation as no model was trained or selected.")

        except (FileNotFoundError, ValueError, RuntimeError, AttributeError) as e:
            error_message = str(e)
            print(f"ERROR during experiment for {self.config['dataset_path']}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e_generic:
            error_message = str(e_generic)
            print(f"UNEXPECTED ERROR for {self.config['dataset_path']}: {e_generic}")
            import traceback; traceback.print_exc()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        result = {
            "dataset_path": self.config['dataset_path'],
            "accuracy": evaluation_metrics.get("accuracy", 0.0),
            "f1_score": evaluation_metrics.get("f1", 0.0),
            "precision": evaluation_metrics.get("precision", 0.0),
            "recall": evaluation_metrics.get("recall", 0.0),
            "confusion_matrix": evaluation_metrics.get("confusion_matrix", None),
            "time": elapsed_time,
            "requested_episodes": self.n_training_episodes,
            "actual_episodes_run": actual_episodes_run,
            "optimal_val_episode": optimal_val_episode,
            "best_val_accuracy": best_val_acc_from_training,
            "backbone": self.backbone_name,
            "error": error_message
        }
        print(f"\n--- Executor: Experiment Summary for {os.path.basename(self.config['dataset_path'])} ---")
        print(f"    Final Test Accuracy: {result['accuracy']:.2f}%")
        print(f"    Time: {result['time']:.2f} seconds")
        if error_message: print(f"    Error during run: {error_message}")
        return result