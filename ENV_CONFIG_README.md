# Environment Configuration Guide

## Overview
All hardcoded values from `main.py` and `task_processor.py` have been moved to environment variables for better configurability. The configuration is managed through the `config.env` file.

## Intelligent Sampling
The system now uses intelligent sampling based on few-shot learning requirements. Instead of fixed sampling values, it calculates the number of files to download based on:
- **For files_per_writer**: `(n_shot + n_query) * sampling_multiplier`
- **For percentage**: Uses `sampling_multiplier` as the percentage directly

This ensures adequate samples for proper episode formation while avoiding excessive data downloads.

### Why Intelligent Sampling?
1. **Episode-Aware**: Sampling is directly related to the few-shot learning requirements
2. **Adaptive**: Automatically adjusts when you change N_SHOT or N_QUERY values
3. **Efficient**: Downloads just enough data for proper training, not arbitrary amounts
4. **Consistent**: Same multiplier provides consistent data availability across different N-way configurations

## Configuration File
The system uses `config.env` file for configuration. This file contains all configurable parameters with their default values.

## Environment Variables

### Dataset and Model Paths
```bash
BASE_DATASET_PATH=D:\IIT\Final Year Project\2025 Project\Final Datasets\Datasets
DATASET_NAMES=Combined_Balanced_All_Equal_Groups  # Comma-separated list
MODEL_SAVE_PATH=D:\IIT\Final Year Project\2025 Project\IPD\Code\Models
EMBEDDINGS_SAVE_PATH=D:\IIT\Final Year Project\2025 Project\IPD\Code\Embeddings
TEMP_DATA_PATH=./temp_data
```

### Dataset Sampling Configuration
```bash
SAMPLING_STRATEGY=files_per_writer  # Options: files_per_writer, percentage
SAMPLING_MULTIPLIER=3.0             # Multiplier for (n_shot + n_query) calculation OR percentage (0.0-1.0)
SAMPLING_SEED=42                    # Random seed for reproducible sampling
```

### Model Training Configuration
```bash
N_WAY=5                    # Number of classes per episode
N_SHOT=5                   # Number of support samples per class
N_QUERY=5                  # Number of query samples per class
N_EVALUATION_TASKS=1000    # Number of evaluation tasks (main.py)
N_TRAINING_EPISODES=10     # Training episodes for task_processor.py
MAX_N_TRAIN_EPISODES=10000 # Max training episodes for main.py
LEARNING_RATE=0.0001       # Learning rate
SEED=42                    # Random seed
BACKBONE_NAME=googlenet    # Backbone network architecture
PRETRAINED_BACKBONE=true   # Use pretrained backbone (true/false)
EVALUATION_INTERVAL=600    # Evaluation interval
EARLY_STOPPING_PATIENCE=5  # Early stopping patience
```

### Data Processing Configuration
```bash
IMAGE_SIZE=224               # Input image size
TRAIN_RATIO=0.7             # Training/testing split ratio
NUM_WORKERS_MAIN=3          # Number of workers for main.py
NUM_WORKERS_TASK_PROCESSOR=0 # Number of workers for task_processor.py
```

### Upload Configuration
```bash
MAX_UPLOAD_RETRIES=3         # Maximum upload retry attempts
UPLOAD_RETRY_DELAY=5         # Delay between retries (seconds)
UPLOAD_TIMEOUT=120           # Upload timeout (seconds)
CHUNK_SIZE_MB=4              # Chunk size for large file uploads (MB)
LARGE_FILE_THRESHOLD_MB=10   # Threshold for chunked upload (MB)
```

### File Names
```bash
RESULTS_BLOB_NAME=training_results.json
MODEL_BLOB_NAME=best_model.pth
ANALYSIS_BLOB_NAME=analysis-results.json
FINAL_MODEL_NAME=final_model.pth
```

### API Configuration (Required for Status Tracking)
```bash
API_BASE_URL=https://localhost:44302       # Base URL for portal API
API_KEY=your_api_key_here                 # Authentication key for API calls (REQUIRED)
MODEL_API_AUTH_METHOD=auto                # Model API authentication method
```

**Important**: The `API_KEY` is required for automatic dataset and model status tracking. If not configured:
- System will show warnings but continue to function
- No status updates will be sent to the portal
- Training will proceed normally without API integration

#### Authentication Methods for Model API:
- **auto** (default): Try both authentication methods, fallback automatically
- **x-api-key**: Use x-api-key header (same as dataset API)
- **bearer**: Use Authorization: Bearer header

**Note**: Different API endpoints may require different authentication methods. The 'auto' setting will try both and use whichever works.

## Usage

### 1. Modify Configuration
Edit the `config.env` file to change any parameters:
```bash
# Example: Increase sampling multiplier for more samples per writer
SAMPLING_MULTIPLIER=4.0  # (n_shot + n_query) * 4.0

# Example: Use ResNet backbone
BACKBONE_NAME=resnet50

# Example: Change learning rate
LEARNING_RATE=0.001
```

### 2. Multiple Dataset Names
To process multiple datasets, use comma-separated values:
```bash
DATASET_NAMES=Dataset1,Dataset2,Dataset3
```

### 3. Percentage-based Sampling
To use percentage-based sampling instead of files per writer:
```bash
SAMPLING_STRATEGY=percentage
SAMPLING_MULTIPLIER=0.25  # Download 25% of all files
```

### 4. Environment Variables Override
You can also set environment variables directly in your system or in your IDE:
```bash
export LEARNING_RATE=0.001
export N_WAY=10
export BACKBONE_NAME=resnet50
export SAMPLING_MULTIPLIER=2.5
```

## Sampling Calculation Examples

### Files Per Writer Strategy
With `N_SHOT=5`, `N_QUERY=5`, and `SAMPLING_MULTIPLIER=3.0`:
- **Calculation**: (5 + 5) × 3.0 = 30 files per writer
- **Reasoning**: Each episode needs 5 support + 5 query = 10 samples per writer. With 3× multiplier, we download 30 files to provide variety across multiple episodes.

### Different Scenarios
```bash
# Conservative sampling (2× minimum required)
N_SHOT=5, N_QUERY=5, SAMPLING_MULTIPLIER=2.0 → 20 files per writer

# Balanced sampling (3× minimum required) - Default
N_SHOT=5, N_QUERY=5, SAMPLING_MULTIPLIER=3.0 → 30 files per writer

# Rich sampling (4× minimum required)
N_SHOT=5, N_QUERY=5, SAMPLING_MULTIPLIER=4.0 → 40 files per writer

# Higher shot/query requirements
N_SHOT=10, N_QUERY=10, SAMPLING_MULTIPLIER=2.0 → 40 files per writer
```

## Files Modified

### 1. `entry_points/main.py`
- Added dotenv import and configuration loading
- Replaced all hardcoded values with `os.getenv()` calls
- Added type conversion (int, float, bool) for environment variables
- Added support for multiple dataset names

### 2. `core/task_processor.py`
- Added dotenv import and configuration loading
- Updated constructor to load sampling strategy from environment
- Modified `train_model()` method to use environment variables
- Updated all blob names and upload parameters to use environment variables
- Modified `_chunked_upload()` method to use configurable chunk size

### 3. `config.env`
- New configuration file containing all configurable parameters
- Well-organized sections for different types of configuration
- Default values matching the original hardcoded values

## Backwards Compatibility
All environment variables have fallback default values that match the original hardcoded values, ensuring the system works without modification if no `config.env` file is present.

## Model Status Tracking
The system now includes automatic model status tracking through API calls. When training starts from a queue message:

1. **Model ID Extraction**: Extracts ID from container name (e.g., `model-1` → `1`)
2. **Status Updates**: Automatically updates model status:
   - `Processing (1)` - When training starts
   - `Completed (2)` - When training succeeds
   - `Failed (3)` - When training fails
3. **API Integration**: Makes GET calls to `api/external/models/{id}/status` and PUT calls to `api/external/models/status` with ModelId in body

See `MODEL_STATUS_TRACKING.md` for detailed documentation.

## Benefits
1. **Easy Configuration**: Change parameters without modifying code
2. **Environment-specific Settings**: Different configurations for dev/test/prod
3. **Version Control**: Keep sensitive configurations out of code
4. **Flexibility**: Override individual parameters as needed
5. **Maintainability**: Centralized configuration management
6. **Automatic Status Tracking**: Real-time model training status updates

## Example Configurations

### High-Performance Setup
```bash
NUM_WORKERS_MAIN=8
NUM_WORKERS_TASK_PROCESSOR=4
CHUNK_SIZE_MB=8
MAX_UPLOAD_RETRIES=5
```

### Quick Testing Setup
```bash
SAMPLING_STRATEGY=percentage
SAMPLING_MULTIPLIER=0.1      # Only 10% of data
MAX_N_TRAIN_EPISODES=100     # Fewer episodes
N_EVALUATION_TASKS=50        # Fewer evaluation tasks
```

### Conservative Sampling Setup
```bash
SAMPLING_STRATEGY=files_per_writer
SAMPLING_MULTIPLIER=2.0      # 2× (n_shot + n_query) - minimal but sufficient
N_SHOT=3
N_QUERY=3                    # Results in 12 files per writer
```

### Large-scale Training Setup
```bash
SAMPLING_STRATEGY=files_per_writer
SAMPLING_MULTIPLIER=5.0      # 5× (n_shot + n_query) - rich diversity
MAX_N_TRAIN_EPISODES=50000   # More training episodes
N_EVALUATION_TASKS=5000      # More evaluation tasks
LEARNING_RATE=0.0005         # Lower learning rate
``` 