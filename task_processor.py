import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from azure.storage.blob import BlobServiceClient
from collections import defaultdict
import json
import torch
from executor import TaskExecutor
import shutil
import time
import uuid

class TaskProcessor:
    def __init__(self, connection_string):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def analyze_dataset(self, container_name):
        """
        Analyzes a dataset directly from an Azure Blob Storage container.

        The analysis includes:
        - Number of writers
        - Writer names
        - Minimum sample count
        - Maximum sample count
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            writer_counts = defaultdict(int)
            blobs = container_client.list_blobs()
            
            for blob in blobs:
                if blob.name.endswith('.json'):
                    continue
                parts = blob.name.split('/')
                if len(parts) > 1:
                    writer_id = parts[0]
                    writer_counts[writer_id] += 1

            if not writer_counts:
                return {
                    "message": "No analyzable data found in the container."
                }
            
            num_writers = len(writer_counts)
            writer_names = list(writer_counts.keys())
            min_samples = min(writer_counts.values())
            max_samples = max(writer_counts.values())

            analysis = {
                "num_writers": num_writers,
                "writer_names": writer_names,
                "min_samples": min_samples,
                "max_samples": max_samples,
                "writer_counts": dict(writer_counts)
            }

            analysis_blob_name = f"{container_name}-analysis.json"
            analysis_blob_client = container_client.get_blob_client(analysis_blob_name)
            analysis_json = json.dumps(analysis, indent=4)
            
            analysis_blob_client.upload_blob(analysis_json, overwrite=True)
            print(f"Uploaded analysis to {container_name}/{analysis_blob_name}")

            return analysis
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None 

    def train_model(self, dataset_container_name, model_container_name):
        """
        Trains a model using the specified dataset and saves the results.
        """
        try:
            print(f"Starting model training with dataset from '{dataset_container_name}'...")

            local_dataset_path = f"./temp_data/{dataset_container_name}"
            print("\n--- PyTorch & CUDA Diagnostics ---")
            print(f"PyTorch version: {torch.__version__}")
            is_cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {is_cuda_available}")
            if is_cuda_available:
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                print("CUDA not available. Training will use CPU.")
                print("Things to check:")
                print("1. Is PyTorch installed with CUDA support? (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")
                print("2. Are NVIDIA drivers installed correctly?")
                print("3. Is the CUDA Toolkit version compatible with your PyTorch and NVIDIA driver versions?")
            print("-------------------------------------\\n")

            run_config = {
                'dataset_path': local_dataset_path,
                'image_size': 224,
                'train_ratio': 0.7,
                'num_workers': 0,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            print(run_config)

            models_dir = os.path.join(os.path.dirname(local_dataset_path), "models")
            os.makedirs(models_dir, exist_ok=True)
            model_save_path = models_dir

            executor = TaskExecutor(
                run_config=run_config,
                n_way=5,
                n_shot=5,
                n_query=5,
                n_training_episodes=10,
                n_evaluation_tasks=10,
                learning_rate=0.001,
                backbone_name="googlenet",
                pretrained_backbone=True,
                seed=42,
                evaluation_interval=600,
                early_stopping_patience=5,
                model_save_path=model_save_path
            )

            result = executor.run_single_experiment()
            
            model_path = None
            
            if 'model_path' in result and result['model_path'] and os.path.exists(result['model_path']):
                model_path = result['model_path']
                print(f"Found model file from result: {model_path}")
            
            if not model_path:
                print(f"Searching for model files in: {models_dir}")
                for root, dirs, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            potential_path = os.path.join(root, file)
                            if os.path.exists(potential_path):
                                model_path = potential_path
                                print(f"Found model file during search: {model_path}")
                                break
                    if model_path:
                        break
            
            if not model_path:
                print("No model file found. Saving current model state...")
                try:
                    final_model_path = os.path.join(models_dir, "final_model.pth")
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(executor.proto_model.state_dict(), final_model_path)
                    if os.path.exists(final_model_path):
                        model_path = final_model_path
                        print(f"Successfully saved final model at: {model_path}")
                    else:
                        print("Failed to save final model")
                except Exception as e:
                    print(f"Error saving final model: {e}")
            
            if 'confusion_matrix' in result and hasattr(result['confusion_matrix'], 'tolist'):
                result['confusion_matrix'] = result['confusion_matrix'].tolist()

            results_json = json.dumps(result, indent=4)

            model_container_client = self.blob_service_client.get_container_client(model_container_name)
            try:
                model_container_client.create_container()
            except Exception as e:
                if "ContainerAlreadyExists" not in str(e):
                    raise
            
            results_blob_name = "training_results.json"
            model_container_client.upload_blob(name=results_blob_name, data=results_json, overwrite=True)
            print(f"Training results uploaded to {model_container_name}/{results_blob_name}")
            
            if model_path and os.path.exists(model_path):
                print(f"Preparing to upload model from: {model_path}")
                
                file_size = os.path.getsize(model_path)
                print(f"Model file size: {file_size / (1024*1024):.2f} MB")
                
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        print(f"Upload attempt {attempt + 1}/{max_retries}")
                        model_blob_name = "best_model.pth"
                        
                        blob_client = model_container_client.get_blob_client(model_blob_name)
                        
                        if file_size > 10 * 1024 * 1024:
                            print("Using chunked upload for large file...")
                            self._chunked_upload(blob_client, model_path)
                        else:
                            print("Using standard upload...")
                            with open(model_path, "rb") as data:
                                blob_client.upload_blob(
                                    data, 
                                    overwrite=True,
                                    timeout=120
                                )
                        
                        print(f"Model file uploaded to {model_container_name}/{model_blob_name}")
                        break
                        
                    except Exception as e:
                        print(f"Upload attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print("All upload attempts failed. Model file not uploaded.")
            else:
                print(f"Warning: No model file available to upload. model_path: {model_path}")
                print(f"Models directory contents: {list(os.listdir(models_dir)) if os.path.exists(models_dir) else 'Directory does not exist'}")
            

            
            return result

        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return None 

    def _chunked_upload(self, blob_client, file_path, chunk_size=4*1024*1024):
        """Upload a file in chunks for better reliability with large files."""
        try:
            try:
                blob_client.delete_blob()
            except:
                pass
            
            block_list = []
            
            with open(file_path, 'rb') as file:
                chunk_num = 0
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    
                    block_id = str(uuid.uuid4())
                    
                    blob_client.stage_block(block_id, chunk, timeout=60)
                    block_list.append(block_id)
                    
                    chunk_num += 1
                    print(f"Uploaded chunk {chunk_num} ({len(chunk)} bytes)")
            
            print("Committing all chunks...")
            blob_client.commit_block_list(block_list, timeout=60)
            print("Chunked upload completed successfully!")
            
        except Exception as e:
            print(f"Chunked upload failed: {e}")
            raise 