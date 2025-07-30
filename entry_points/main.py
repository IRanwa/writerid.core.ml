import torch
import random
import numpy as np
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config.env file
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment configuration
load_dotenv(os.path.join(project_root, 'config.env'))

from core.executor import TaskExecutor
from utils.plotting_utils import plot_results

config = {}
def set_seeds(seed_value: int, device_type: str):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Load configuration from environment variables with fallback defaults
    BASE_DATASET_PATH = os.getenv('BASE_DATASET_PATH', r"D:\IIT\Final Year Project\2025 Project\Final Datasets\Datasets")
    DATASET_NAMES = os.getenv('DATASET_NAMES', 'Combined_Balanced_All_Equal_Groups').split(',')
    
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', r"D:\IIT\Final Year Project\2025 Project\IPD\Code\Models")
    EMBEDDINGS_SAVE_PATH = os.getenv('EMBEDDINGS_SAVE_PATH', r"D:\IIT\Final Year Project\2025 Project\IPD\Code\Embeddings")
    
    base_run_config = {
        'image_size': int(os.getenv('IMAGE_SIZE', '224')),
        'train_ratio': float(os.getenv('TRAIN_RATIO', '0.7')),
        'num_workers': int(os.getenv('NUM_WORKERS_MAIN', '3')),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    N_WAY = int(os.getenv('N_WAY', '5'))
    N_SHOT = int(os.getenv('N_SHOT', '5'))
    N_QUERY = int(os.getenv('N_QUERY', '5'))
    N_EVALUATION_TASKS = int(os.getenv('N_EVALUATION_TASKS', '1000'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
    SEED = int(os.getenv('SEED', '42'))
    BACKBONE_TO_USE = os.getenv('BACKBONE_NAME', 'googlenet')
    MAX_N_TRAIN_EPISODES = int(os.getenv('MAX_N_TRAIN_EPISODES', '10000'))
    EVALUATION_INTERVAL = int(os.getenv('EVALUATION_INTERVAL', '600'))
    EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', '5'))
    PRETRAINED_BACKBONE = os.getenv('PRETRAINED_BACKBONE', 'true').lower() == 'true'
    
    set_seeds(SEED, base_run_config['device'])

    all_experiment_results = []

    print(f"Starting experiments - Device: {base_run_config['device']}, Backbone: {BACKBONE_TO_USE}")

    for dataset_name in DATASET_NAMES:
        current_dataset_path = os.path.join(BASE_DATASET_PATH, dataset_name)
        print(f"\nProcessing dataset: {dataset_name}")
        
        run_specific_config = base_run_config.copy()
        run_specific_config['dataset_path'] = current_dataset_path        
        
        executor = TaskExecutor(
            run_config=run_specific_config,
            n_way=N_WAY,
            n_shot=N_SHOT,
            n_query=N_QUERY,
            n_training_episodes=MAX_N_TRAIN_EPISODES,
            n_evaluation_tasks=N_EVALUATION_TASKS,
            learning_rate=LEARNING_RATE,
            backbone_name=BACKBONE_TO_USE,
            pretrained_backbone=PRETRAINED_BACKBONE,
            seed=SEED,
            evaluation_interval=EVALUATION_INTERVAL,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            model_save_path=MODEL_SAVE_PATH,
            embeddings_save_path=EMBEDDINGS_SAVE_PATH 
        )

        try:
            result = executor.run_single_experiment()
            
            result['dataset_path'] = current_dataset_path
            result['dataset_name_for_plot'] = dataset_name
            result['requested_episodes'] = MAX_N_TRAIN_EPISODES
            result['backbone'] = BACKBONE_TO_USE
            all_experiment_results.append(result)
            print(f"Experiment completed - Episodes: {result.get('actual_episodes_run', 'N/A')}")
        except Exception as e:
            print(f"ERROR in experiment: {e}")
            import traceback
            traceback.print_exc()
            all_experiment_results.append({
                "dataset_path": current_dataset_path,
                "dataset_name_for_plot": dataset_name,
                "accuracy": 0.0, "f1_score":0.0, "precision":0.0, "recall":0.0, "confusion_matrix": None, "time": 0.0,
                "requested_episodes": MAX_N_TRAIN_EPISODES,
                "actual_episodes_run": 0,
                "optimal_val_episode": 0,
                "best_val_accuracy": 0.0,
                "backbone": BACKBONE_TO_USE, 
                "error": str(e)
            })
        
    print('\n=== EXPERIMENT RESULTS ===')
    for res in all_experiment_results:
        dataset_name = res.get('dataset_name_for_plot', 'Unknown')
        print(f"\nDataset: {dataset_name}")
        print(f"  Episodes Run: {res.get('actual_episodes_run', 'N/A')}")
        if res.get('optimal_val_episode', 0) > 0:
             print(f"  Best Validation: {res.get('best_val_accuracy', 0.0):.2f}% at episode {res.get('optimal_val_episode')}")
        print(f"  Test Accuracy: {res.get('accuracy', 0.0):.2f}%")
        print(f"  F1 Score: {res.get('f1_score', 0.0):.2f}%")
        print(f"  Precision: {res.get('precision', 0.0):.2f}%")
        print(f"  Recall: {res.get('recall', 0.0):.2f}%")
        print(f"  Time: {res.get('time', 0.0):.2f}s")
        
        confusion_matrix = res.get('confusion_matrix')
        if confusion_matrix is not None:
            print(f"  Confusion Matrix:")
            for i, row in enumerate(confusion_matrix):
                print(f"    Class {i}: {row}")
        else:
            print(f"  Confusion Matrix: Not available")
            
        if res.get('error'):
            print(f"  Error: {res.get('error')}")

    if all_experiment_results:
        print("\nGenerating plots...")
        plot_results(all_experiment_results, plot_type="all_metrics_per_dataset")
        
        for res_item in all_experiment_results:
            if res_item and not res_item.get('error'):
                plot_results([res_item], plot_type="confusion_matrix", n_way_for_cm_plot=N_WAY)
            elif res_item.get('error'):
                dataset_name = res_item.get('dataset_name_for_plot', 'Unknown')
                print(f"Skipping confusion matrix for {dataset_name} due to error")
    else:
        print("No results collected")
        
    print("\nExperiments completed")

if __name__ == '__main__':
    main()