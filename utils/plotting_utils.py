import matplotlib.pyplot as plt
import numpy as np
import seaborn
import os
import textwrap
from typing import List, Dict, Optional

def plot_results(
    results_list: List[Dict],
    plot_type: str,
    n_way_for_cm_plot: Optional[int] = None
):
    if not results_list:
        print("Plotting: No results to plot.")
        return

    if plot_type == "all_metrics_per_dataset":
        fig, ax = plt.subplots(figsize=(14, 8))
        results_list.sort(key=lambda x: str(x.get('dataset_path', '')))
        
        dataset_names = [os.path.basename(str(item.get('dataset_path', f'Run {i+1}'))) for i, item in enumerate(results_list)]
        accuracies = [item.get('accuracy', 0) for item in results_list]
        f1_scores = [item.get('f1_score', 0) for item in results_list]
        precisions = [item.get('precision', 0) for item in results_list]
        recalls = [item.get('recall', 0) for item in results_list]
        
        req_episodes = results_list[0].get('requested_episodes', "N/A")
        backbone_used = results_list[0].get('backbone', "N/A")
        
        x_indices = np.arange(len(dataset_names))
        bar_total_width = 0.8
        bar_individual_width = bar_total_width / 4

        ax.bar(x_indices - 1.5 * bar_individual_width, accuracies, bar_individual_width, label='Accuracy', color='skyblue')
        ax.bar(x_indices - 0.5 * bar_individual_width, f1_scores, bar_individual_width, label='F1 (macro)', color='lightcoral')
        ax.bar(x_indices + 0.5 * bar_individual_width, precisions, bar_individual_width, label='Precision (macro)', color='lightgreen')
        ax.bar(x_indices + 1.5 * bar_individual_width, recalls, bar_individual_width, label='Recall (macro)', color='gold')

        for bar_group in ax.containers:
            ax.bar_label(bar_group, fmt='%.2f', fontsize=9, padding=3)

        ax.set_xlabel('Dataset', fontsize=12, labelpad=15)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'Performance Metrics per Dataset (Backbone: {backbone_used}, Episodes: {req_episodes})', fontsize=14)
        ax.set_xticks(x_indices)
        
        wrapped_labels = ['\n'.join(textwrap.wrap(name, 15)) for name in dataset_names]
        ax.set_xticklabels(wrapped_labels, fontsize=10)
        
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()

    elif plot_type == "confusion_matrix":
        if not n_way_for_cm_plot:
            print("Plotting Error: `n_way_for_cm_plot` must be provided for confusion matrix.")
            return
            
        result_for_cm = results_list[0]
        cm = result_for_cm.get('confusion_matrix')
        
        if cm is None:
            print("Plotting Info: No confusion matrix data available for this run.")
            return

        dataset_label = os.path.basename(str(result_for_cm.get('dataset_path', 'Unknown Dataset')))
        
        optimal_ep = result_for_cm.get('optimal_val_episode')
        actual_ep_run = result_for_cm.get('actual_episodes_run')
        requested_ep = result_for_cm.get('requested_episodes', 'N/A')

        episode_display_text = ""
        if optimal_ep and optimal_ep > 0:
            episode_display_text = f"Episode ~{optimal_ep}"
        elif actual_ep_run and actual_ep_run > 0:
            episode_display_text = f"Episode ~{actual_ep_run} (no validation improvement)"
        else:
            episode_display_text = f"Max Episodes {requested_ep} (error or no training)"
        
        if not episode_display_text or episode_display_text == "Episode ~N/A":
             episode_display_text = f"Max Episodes {requested_ep}"

        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        seaborn.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                        xticklabels=range(n_way_for_cm_plot),
                        yticklabels=range(n_way_for_cm_plot), ax=ax_cm)
                        
        ax_cm.set_title(f'Confusion Matrix for {dataset_label}\n(Model from {episode_display_text})', fontsize=14)
        ax_cm.set_xlabel('Predicted Label', fontsize=12)
        ax_cm.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    else:
        print(f"Plotting Error: Unknown plot type '{plot_type}'. Choose 'all_metrics_per_dataset' or 'confusion_matrix'.")