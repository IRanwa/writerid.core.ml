# Dataset Random Sampling Feature

## Overview
The code has been modified to download a random subset of dataset files instead of downloading all files from Azure Blob Storage. This feature helps reduce download time and storage requirements while maintaining dataset diversity.

## Changes Made

### 1. Modified `core/task_processor.py`
- **Added random import**: `import random`
- **Enhanced `_download_dataset()` method** with new parameters:
  - `max_files_per_writer`: Limit the number of files downloaded per writer
  - `download_percentage`: Download a percentage of total files (0.0-1.0)
  - `random_seed`: Seed for reproducible random sampling
- **Added `set_sampling_strategy()` method** for easy configuration
- **Updated constructor** with default sampling settings

### 2. Updated `entry_points/queue_listener.py`
- Added examples and comments showing how to configure sampling strategies

## Usage Examples

### Method 1: Files Per Writer with Multiplier
```python
# Use 4× multiplier for (n_shot + n_query) calculation
task_processor.set_sampling_strategy("files_per_writer", 4.0, seed=42)
# With N_SHOT=5, N_QUERY=5: downloads (5+5)*4.0 = 40 files per writer
```

### Method 2: Percentage-based Sampling
```python
# Download 25% of all available files
task_processor.set_sampling_strategy("percentage", 0.25, seed=42)
```

### Method 3: Direct Parameter Usage
```python
# When calling download directly (calculated values will be used)
success = task_processor._download_dataset(
    container_name="my_container", 
    local_path="./local_data",
    max_files_per_writer=None,  # Will be calculated from n_shot + n_query
    random_seed=42              # Reproducible sampling
)
```

## Default Configuration
- **Strategy**: `files_per_writer`
- **Multiplier**: `3.0` (3× the required n_shot + n_query samples)
- **Seed**: `42` (for reproducible results)

## Intelligent Sampling Calculation
The system now calculates sampling based on few-shot learning requirements:
- **Files per writer**: `(n_shot + n_query) × multiplier`
- **Example**: With N_SHOT=5, N_QUERY=5, MULTIPLIER=3.0 → 30 files per writer
- **Benefit**: Ensures adequate samples for episode formation while avoiding excessive downloads

## Configuration in queue_listener.py
The queue listener includes commented examples showing how to modify the sampling strategy:

```python
# Option 1: Files per writer with intelligent multiplier
# self.task_processor.set_sampling_strategy("files_per_writer", 4.0, seed=42)  # 4× multiplier

# Option 2: Download percentage of total files
# self.task_processor.set_sampling_strategy("percentage", 0.25, seed=42)       # 25% of files
```

## Key Features
1. **Per-writer sampling**: Ensures balanced representation across different writers
2. **Percentage sampling**: Downloads a fixed percentage of total files
3. **Reproducible sampling**: Uses random seed for consistent results
4. **Maintains directory structure**: Preserves original file organization
5. **Detailed logging**: Shows sampling statistics and progress

## Benefits
- **Faster downloads**: Significantly reduced download time
- **Storage efficiency**: Uses less local storage space
- **Balanced datasets**: Maintains writer diversity
- **Reproducible results**: Same sampling with same seed
- **Flexible configuration**: Easy to adjust sampling parameters

## Sampling Output Example
```
Using sampling strategy: files_per_writer
Calculated sampling: (n_shot=5 + n_query=5) * 3.0 = 30 files per writer
Writer writer001: Selected 30 out of 150 files
Writer writer002: Selected 30 out of 200 files
Writer writer003: Selected 30 out of 43 files
Dataset download completed. Downloaded 90 files, skipped 2 files.
Applied per-writer limit: 30 files per writer
```

## Notes
- JSON files (analysis results) are automatically skipped
- If a writer has fewer files than the limit, all files are downloaded
- The sampling respects the original directory structure
- Random seed ensures reproducible sampling across runs 