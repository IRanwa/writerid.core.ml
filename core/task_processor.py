import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from azure.storage.blob import BlobServiceClient
from collections import defaultdict
import json
import torch
from .executor import TaskExecutor
from utils.api_client import ApiClient
import shutil
import time
import uuid
import random
from typing import Dict, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment configuration
project_root = Path(__file__).parent.parent
load_dotenv(os.path.join(project_root, 'config.env'))

class TaskProcessor:
    def __init__(self, connection_string):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.api_client = ApiClient()
        # Load sampling strategy from environment variables with fallback defaults
        self.sampling_strategy = os.getenv('SAMPLING_STRATEGY', 'files_per_writer')
        self.sampling_multiplier = float(os.getenv('SAMPLING_MULTIPLIER', '3.0'))
        self.sampling_seed = int(os.getenv('SAMPLING_SEED', '42'))

    def _get_dataset_status(self, task_id: str) -> Optional[Dict]:
        """Get dataset status from API"""
        return self.api_client.get_dataset_status(task_id)

    def _update_dataset_status(self, dataset_id: str, status: int, message: str = "") -> bool:
        """Update dataset status via API"""
        return self.api_client.update_dataset_status(dataset_id, status, message)

    def set_sampling_strategy(self, strategy: str = "files_per_writer", multiplier: float = 3.0, seed: int = 42):
        """
        Configure the dataset sampling strategy.
        
        Args:
            strategy: 'files_per_writer' or 'percentage'  
            multiplier: Multiplier for (n_shot + n_query) calculation or percentage 0.0-1.0 (for 'percentage')
            seed: Random seed for reproducible sampling
        """
        self.sampling_strategy = strategy
        self.sampling_multiplier = multiplier
        self.sampling_seed = seed
        print(f"Sampling strategy set to: {strategy} with multiplier {multiplier}, seed {seed}")

    def analyze_dataset(self, task_id: str, container_name: str):
        """
        API-based dataset analysis method.
        Processes dataset analysis requests using task ID and container name.
        """
        print(f"Starting dataset analysis for task ID: {task_id}, container: {container_name}")
        
        try:
            analysis_result = self._perform_dataset_analysis(container_name)
            
            if analysis_result:
                success_message = f"Analysis completed successfully. Found {analysis_result.get('num_writers', 0)} writers."
                update_success = self._update_dataset_status(task_id, 2, success_message)
                
                if update_success:
                    print("Dataset analysis completed and API status updated successfully")
                else:
                    print("Analysis completed but failed to update API status")
                    
                return analysis_result
            else:
                error_message = "Dataset analysis failed - no analyzable data found"
                self._update_dataset_status(task_id, 3, error_message)
                return None
                
        except Exception as e:
            error_message = f"Dataset analysis failed: {str(e)}"
            print(error_message)
            self._update_dataset_status(task_id, 3, error_message)
            return None

    def _perform_dataset_analysis(self, container_name: str):
        """
        Performs the actual dataset analysis on the Azure Blob Storage container.
        
        Args:
            container_name: Name of the Azure Blob Storage container
            
        Returns:
            Dict: Analysis results or None if failed
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
                return None
            
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


            analysis_blob_name = os.getenv('ANALYSIS_BLOB_NAME', 'analysis-results.json')
            analysis_blob_client = container_client.get_blob_client(analysis_blob_name)
            analysis_json = json.dumps(analysis, indent=4)
            
            analysis_blob_client.upload_blob(analysis_json, overwrite=True)
            print(f"Uploaded analysis to {container_name}/{analysis_blob_name}")

            return analysis
        
        except Exception as e:
            print(f"An error occurred during dataset analysis: {e}")
            return None

    def _download_dataset(self, container_name: str, local_path: str, 
                         max_files_per_writer: Optional[int] = None, 
                         download_percentage: Optional[float] = None,
                         random_seed: Optional[int] = None) -> bool:
        """
        Downloads a subset of the dataset from Azure Blob Storage to a local directory.
        
        Args:
            container_name: Name of the Azure Blob Storage container
            local_path: Local directory path to download to
            max_files_per_writer: Maximum number of files to download per writer (optional)
            download_percentage: Percentage of total files to download (0.0-1.0, optional)
            random_seed: Random seed for reproducible sampling (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Downloading dataset from container '{container_name}' to local path '{local_path}'...")
            
            # Set random seed if provided
            if random_seed is not None:
                random.seed(random_seed)
            
            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # First, collect all blobs and organize by writer
            blobs = list(container_client.list_blobs())
            writer_files = defaultdict(list)
            json_files = []
            
            for blob in blobs:
                # Skip analysis results and other JSON files but keep track of them
                if blob.name.endswith('.json'):
                    json_files.append(blob)
                    continue
                
                # Organize files by writer (assuming directory structure: writer_id/file)
                parts = blob.name.split('/')
                if len(parts) > 1:
                    writer_id = parts[0]
                    writer_files[writer_id].append(blob)
                else:
                    # Files not organized by writer - treat as single group
                    writer_files['root'].append(blob)
            
            # Select files to download based on specified criteria
            files_to_download = []
            
            if max_files_per_writer is not None:
                # Limit files per writer
                for writer_id, files in writer_files.items():
                    if len(files) <= max_files_per_writer:
                        files_to_download.extend(files)
                    else:
                        # Randomly sample files for this writer
                        selected_files = random.sample(files, max_files_per_writer)
                        files_to_download.extend(selected_files)
                        print(f"Writer {writer_id}: Selected {max_files_per_writer} out of {len(files)} files")
                        
            elif download_percentage is not None:
                # Download a percentage of total files
                all_files = [file for files in writer_files.values() for file in files]
                num_files_to_download = int(len(all_files) * download_percentage)
                if num_files_to_download == 0:
                    num_files_to_download = 1  # Download at least one file
                files_to_download = random.sample(all_files, min(num_files_to_download, len(all_files)))
                print(f"Selected {len(files_to_download)} out of {len(all_files)} files ({download_percentage*100:.1f}%)")
                
            else:
                # Default: download all files (original behavior)
                files_to_download = [file for files in writer_files.values() for file in files]
            
            downloaded_count = 0
            skipped_count = len(json_files)  # Count JSON files as skipped
            
            # Download selected files
            for blob in files_to_download:
                # Create local file path maintaining directory structure
                local_file_path = os.path.join(local_path, blob.name)
                
                # Create directory structure if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the blob
                blob_client = container_client.get_blob_client(blob.name)
                
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
                downloaded_count += 1
                
                if downloaded_count % 10 == 0:
                    print(f"Downloaded {downloaded_count} files...")
            
            print(f"Dataset download completed. Downloaded {downloaded_count} files, skipped {skipped_count} files.")
            
            # Show sampling summary
            if max_files_per_writer is not None:
                print(f"Applied per-writer limit: {max_files_per_writer} files per writer")
            elif download_percentage is not None:
                print(f"Applied percentage sampling: {download_percentage*100:.1f}% of total files")
            
            # Verify the download was successful
            if downloaded_count == 0:
                print("Warning: No files were downloaded from the container.")
                return False
            
            # Check if we have the expected directory structure
            if not os.path.exists(local_path):
                print(f"Error: Local dataset path {local_path} does not exist after download.")
                return False
            
            # List the contents of the downloaded dataset
            try:
                contents = os.listdir(local_path)
                print(f"Local dataset contains {len(contents)} directories/files: {contents[:10]}...")  # Show first 10
            except Exception as e:
                print(f"Warning: Could not list contents of {local_path}: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False

    def train_model(self, dataset_container_name, model_container_name):
        """
        Trains a model using the specified dataset and saves the results.
        """
        try:
            print(f"Starting model training with dataset from '{dataset_container_name}'...")

            temp_data_path = os.getenv('TEMP_DATA_PATH', './temp_data')
            local_dataset_path = f"{temp_data_path}/{dataset_container_name}"
            
            # Download dataset from Azure Blob Storage with configured random sampling
            # Calculate sampling based on n_shot and n_query values
            n_shot = int(os.getenv('N_SHOT', '5'))
            n_query = int(os.getenv('N_QUERY', '5'))
            
            # Calculate sampling value based on episode requirements
            if self.sampling_strategy == "files_per_writer":
                # For files_per_writer: calculate based on (n_shot + n_query) * multiplier
                calculated_sampling_value = int((n_shot + n_query) * self.sampling_multiplier)
                print(f"Using sampling strategy: {self.sampling_strategy}")
                print(f"Calculated sampling: (n_shot={n_shot} + n_query={n_query}) * {self.sampling_multiplier} = {calculated_sampling_value} files per writer")
            elif self.sampling_strategy == "percentage":
                # For percentage: use multiplier as percentage directly
                calculated_sampling_value = self.sampling_multiplier
                print(f"Using sampling strategy: {self.sampling_strategy} with {calculated_sampling_value*100:.1f}% of files")
            
            # Configure sampling parameters based on strategy
            max_files_per_writer = None
            download_percentage = None
            
            if self.sampling_strategy == "files_per_writer":
                max_files_per_writer = calculated_sampling_value
            elif self.sampling_strategy == "percentage":
                download_percentage = calculated_sampling_value
            
            if not self._download_dataset(dataset_container_name, local_dataset_path, 
                                        max_files_per_writer=max_files_per_writer,
                                        download_percentage=download_percentage,
                                        random_seed=self.sampling_seed):
                print(f"Failed to download dataset from container: {dataset_container_name}")
                return None
            
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
                'image_size': int(os.getenv('IMAGE_SIZE', '224')),
                'train_ratio': float(os.getenv('TRAIN_RATIO', '0.7')),
                'num_workers': int(os.getenv('NUM_WORKERS_TASK_PROCESSOR', '0')),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            print(run_config)

            models_dir = os.path.join(os.path.dirname(local_dataset_path), "models")
            os.makedirs(models_dir, exist_ok=True)
            model_save_path = models_dir

            executor = TaskExecutor(
                run_config=run_config,
                n_way=int(os.getenv('N_WAY', '5')),
                n_shot=int(os.getenv('N_SHOT', '5')),
                n_query=int(os.getenv('N_QUERY', '5')),
                n_training_episodes=int(os.getenv('N_TRAINING_EPISODES', '10')),
                n_evaluation_tasks=int(os.getenv('N_EVALUATION_TASKS', '10')),
                learning_rate=float(os.getenv('LEARNING_RATE', '0.0001')),
                backbone_name=os.getenv('BACKBONE_NAME', 'googlenet'),
                pretrained_backbone=os.getenv('PRETRAINED_BACKBONE', 'true').lower() == 'true',
                seed=int(os.getenv('SEED', '42')),
                evaluation_interval=int(os.getenv('EVALUATION_INTERVAL', '600')),
                early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE', '5')),
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
                    final_model_name = os.getenv('FINAL_MODEL_NAME', 'final_model.pth')
                    final_model_path = os.path.join(models_dir, final_model_name)
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
            
            results_blob_name = os.getenv('RESULTS_BLOB_NAME', 'training_results.json')
            model_container_client.upload_blob(name=results_blob_name, data=results_json, overwrite=True)
            print(f"Training results uploaded to {model_container_name}/{results_blob_name}")
            
            if model_path and os.path.exists(model_path):
                print(f"Preparing to upload model from: {model_path}")
                
                file_size = os.path.getsize(model_path)
                print(f"Model file size: {file_size / (1024*1024):.2f} MB")
                
                max_retries = int(os.getenv('MAX_UPLOAD_RETRIES', '3'))
                retry_delay = int(os.getenv('UPLOAD_RETRY_DELAY', '5'))
                large_file_threshold = int(os.getenv('LARGE_FILE_THRESHOLD_MB', '10')) * 1024 * 1024
                
                for attempt in range(max_retries):
                    try:
                        print(f"Upload attempt {attempt + 1}/{max_retries}")
                        model_blob_name = os.getenv('MODEL_BLOB_NAME', 'best_model.pth')
                        
                        blob_client = model_container_client.get_blob_client(model_blob_name)
                        
                        if file_size > large_file_threshold:
                            print("Using chunked upload for large file...")
                            self._chunked_upload(blob_client, model_path)
                        else:
                            print("Using standard upload...")
                            upload_timeout = int(os.getenv('UPLOAD_TIMEOUT', '120'))
                            with open(model_path, "rb") as data:
                                blob_client.upload_blob(
                                    data, 
                                    overwrite=True,
                                    timeout=upload_timeout
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
        finally:
            # Clean up temporary dataset directory
            if os.path.exists(local_dataset_path):
                try:
                    shutil.rmtree(local_dataset_path)
                    print(f"Cleaned up temporary dataset directory: {local_dataset_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up {local_dataset_path}: {cleanup_error}") 

    def _chunked_upload(self, blob_client, file_path, chunk_size=None):
        """Upload a file in chunks for better reliability with large files."""
        try:
            if chunk_size is None:
                chunk_size = int(os.getenv('CHUNK_SIZE_MB', '4')) * 1024 * 1024
                
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