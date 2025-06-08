import torch
import random
import numpy as np
import os
from executor import TaskExecutor
from plotting_utils import plot_results

config = {}
def set_seeds(seed_value: int, device_type: str):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device_type == 'cuda':
        print("Setting CUDA seeds and options for reproducibility")
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    BASE_DATASET_PATH = r"D:\\IIT\\Final Year Project\\2025 Project\\Final Datasets\\Datasets"
    DATASET_NAMES = ["Combined_Balanced_All_Equal_Groups"]
    
    MODEL_SAVE_PATH = r"D:\\IIT\\Final Year Project\\2025 Project\\IPD\\Code\\Models"
    EMBEDDINGS_SAVE_PATH = r"D:\\IIT\\Final Year Project\\2025 Project\\IPD\\Code\\Embeddings"
    
    base_run_config = {
        'image_size': 224,
        'train_ratio': 0.7,
        'num_workers': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    N_WAY = 5
    N_SHOT = 7
    N_QUERY = 5
    N_EVALUATION_TASKS = 1000
    LEARNING_RATE = 0.0001
    SEED = 42
    BACKBONE_TO_USE = "googlenet"
    MAX_N_TRAIN_EPISODES = 10000
    EVALUATION_INTERVAL = 600
    EARLY_STOPPING_PATIENCE = 5
    
    set_seeds(SEED, base_run_config['device'])

    all_experiment_results = []

    print(f"Using backbone: {BACKBONE_TO_USE}")
    print(f"Device: {base_run_config['device']}")
    print(f"Max Training Episodes per dataset: {MAX_N_TRAIN_EPISODES}")
    print(f"Evaluation Tasks per dataset: {N_EVALUATION_TASKS}")
    print(f"Early Stopping: Interval={EVALUATION_INTERVAL} episodes, Patience={EARLY_STOPPING_PATIENCE}.")

    for dataset_name in DATASET_NAMES:
        current_dataset_path = os.path.join(BASE_DATASET_PATH, dataset_name)
        print(f"\n-- Processing Dataset: {dataset_name} --")
        print(f"Dataset path: {current_dataset_path}")

        run_specific_config = base_run_config.copy()
        run_specific_config['dataset_path'] = current_dataset_path        
        print(f"\n-- Starting Experiment for {dataset_name}: Max Training Episodes: {MAX_N_TRAIN_EPISODES}, Backbone: {BACKBONE_TO_USE} --")
        
        executor = TaskExecutor(
            run_config=run_specific_config,
            n_way=N_WAY,
            n_shot=N_SHOT,
            n_query=N_QUERY,
            n_training_episodes=MAX_N_TRAIN_EPISODES,
            n_evaluation_tasks=N_EVALUATION_TASKS,
            learning_rate=LEARNING_RATE,
            backbone_name=BACKBONE_TO_USE,
            pretrained_backbone=True,
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
        except Exception as e:
            print(f"ERROR during experiment for {dataset_name} (Max Episodes: {MAX_N_TRAIN_EPISODES}): {e}")
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
        actual_episodes_run_display = 'N/A'
        if 'result' in locals() and result:
             actual_episodes_run_display = result.get('actual_episodes_run', 'N/A')
        elif all_experiment_results and all_experiment_results[-1].get('error'): 
             actual_episodes_run_display = all_experiment_results[-1].get('actual_episodes_run', 'N/A')

        print(f"-- Experiment finished for {dataset_name}: Max Training Episodes: {MAX_N_TRAIN_EPISODES} (Actual: {actual_episodes_run_display}) --\n")
        
    print('\n-- All Collected Results --')
    for res_idx, res in enumerate(all_experiment_results):
        dataset_display_name = res.get('dataset_name_for_plot', os.path.basename(str(res.get('dataset_path','Unknown'))))
        print(f"\n--- Result for Dataset: {dataset_display_name} (Requested Max Episodes: {res.get('requested_episodes')}, Backbone: {res.get('backbone')}) ---")
        print(f"  Actual Episodes Run: {res.get('actual_episodes_run', 'N/A')}")
        if res.get('optimal_val_episode', 0) > 0 :
             print(f"  Best Validation Accuracy during training: {res.get('best_val_accuracy', 0.0):.2f}% at episode ~{res.get('optimal_val_episode')}")
        else:
             print("  No validation improvement recorded or early stopping not effective/not reached.")
        print(f"  Final Test Accuracy:         {res.get('accuracy', 0.0):.2f}%")
        print(f"  Final Test F1 Score (macro): {res.get('f1_score', 0.0):.2f}%")
        print(f"  Final Test Precision (macro):{res.get('precision', 0.0):.2f}%")
        print(f"  Final Test Recall (macro):   {res.get('recall', 0.0):.2f}%")
        print(f"  Total Time:                  {res.get('time', 0.0):.2f}s")
        if res.get('error'):
            print(f"  Error: {res.get('error')}")

    if all_experiment_results:
        print("\n--- Generating Overall Performance Metrics Plot ---")
        plot_results(
            all_experiment_results,
            plot_type="all_metrics_per_dataset" 
        )
        
        print("\n--- Generating Individual Confusion Matrices ---")
        for res_item in all_experiment_results:
            if res_item and not res_item.get('error'):
                plot_results([res_item], plot_type="confusion_matrix", n_way_for_cm_plot=N_WAY)
            elif res_item.get('error'):
                dataset_display_name = res_item.get('dataset_name_for_plot', 'Unknown Dataset')
                print(f"Skipping confusion matrix for {dataset_display_name} due to error during experiment.")

    else:
        print("No results were collected to plot.")
        
    print("--- Main Script Finished ---")

if __name__ == '__main__':
    main()