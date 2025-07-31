# WriterID Core ML

A machine learning service for writer identification using few-shot learning with ProtoNet architecture. This service provides dataset analysis, model training, and evaluation capabilities for handwriting recognition tasks.

## Features

- **Dataset Analysis**: Automated analysis of handwriting datasets with writer statistics
- **Few-Shot Learning**: ProtoNet-based model training for writer identification
- **Model Training**: End-to-end training pipeline with early stopping and validation
- **API Integration**: RESTful API integration for status tracking and management
- **Azure Integration**: Seamless integration with Azure Blob Storage and Queue services
- **Flexible Sampling**: Configurable dataset sampling strategies for optimal training

## Project Structure

```
WriterID Core ML/
├── core/                    # Core ML functionality
│   ├── executor.py         # Training execution engine
│   ├── sampler.py          # Dataset sampling utilities
│   └── task_processor.py   # Task processing and API integration
├── data/                   # Data management
│   ├── data_manager.py     # Dataset loading and preprocessing
│   └── image_processor.py  # Image processing utilities
├── entry_points/           # Application entry points
│   ├── main.py            # Main training script
│   └── queue_listener.py  # Azure Queue listener
├── models/                 # Model architectures
│   ├── models.py          # ProtoNet model implementation
│   └── network_architectures.py  # Backbone network definitions
├── utils/                  # Utility functions
│   ├── api_client.py      # API communication utilities
│   ├── plotting_utils.py  # Visualization and plotting
│   └── training_utils.py  # Training and evaluation utilities
├── config.env             # Environment configuration
├── requirements.txt        # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Azure Storage Account (for cloud integration)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd WriterID-Core-ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `config.env.example` to `config.env`
   - Update configuration values in `config.env`

## Configuration

### Environment Variables

Create a `config.env` file with the following variables:

```env
# Dataset Configuration
BASE_DATASET_PATH=D:\IIT\Final Year Project\2025 Project\Final Datasets\Datasets
DATASET_NAMES=Combined_Balanced_All_Equal_Groups

# Model Configuration
BACKBONE_NAME=googlenet
PRETRAINED_BACKBONE=true
IMAGE_SIZE=224
N_WAY=5
N_SHOT=5
N_QUERY=5

# Training Configuration
MAX_N_TRAIN_EPISODES=10000
EVALUATION_INTERVAL=600
EARLY_STOPPING_PATIENCE=5
LEARNING_RATE=0.0001
SEED=42

# Sampling Configuration
SAMPLING_STRATEGY=files_per_writer
SAMPLING_MULTIPLIER=3.0
SAMPLING_SEED=42

# Storage Paths
MODEL_SAVE_PATH=D:\IIT\Final Year Project\2025 Project\IPD\Code\Models
EMBEDDINGS_SAVE_PATH=D:\IIT\Final Year Project\2025 Project\IPD\Code\Embeddings
TEMP_DATA_PATH=./temp_data

# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here

# API Configuration
API_BASE_URL=https://localhost:44302
API_KEY=your_api_key_here
MODEL_API_AUTH_METHOD=auto

# System Configuration
NUM_WORKERS_MAIN=3
TRAIN_RATIO=0.7
N_EVALUATION_TASKS=1000
```

## Usage

### 1. Direct Training

Run the main training script:

```bash
python entry_points/main.py
```

This will:
- Load datasets from the configured path
- Train ProtoNet models on each dataset
- Generate evaluation metrics and confusion matrices
- Save trained models and embeddings

### 2. Queue-Based Processing

Start the queue listener for automated processing:

```bash
python entry_points/queue_listener.py
```

This service will:
- Listen for dataset analysis requests
- Process training tasks from Azure Queue
- Update model and dataset status via API

### 3. Dataset Analysis

Analyze a dataset programmatically:

```python
from core.task_processor import TaskProcessor

processor = TaskProcessor(connection_string="your_azure_connection_string")
result = processor.analyze_dataset("task_id", "container_name")
```

## Model Architecture

### ProtoNet Implementation

The system uses a ProtoNet architecture with the following components:

- **Feature Extractor**: Pre-trained backbone networks (GoogLeNet, ResNet, etc.)
- **Prototype Computation**: Mean embedding calculation for each class
- **Distance Metric**: Euclidean distance for similarity computation
- **Classification**: Nearest prototype assignment

### Supported Backbones

- GoogLeNet (default)
- ResNet variants
- EfficientNet variants
- Custom architectures

## Training Process

1. **Episode Generation**: Create few-shot episodes with support and query sets
2. **Prototype Computation**: Calculate class prototypes from support set
3. **Distance Calculation**: Compute distances between query samples and prototypes
4. **Loss Computation**: Cross-entropy loss on distance-based scores
5. **Optimization**: Adam optimizer with learning rate scheduling
6. **Validation**: Regular evaluation on validation episodes
7. **Early Stopping**: Stop training when validation performance plateaus

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro-averaged F1 score
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **Confusion Matrix**: Detailed class-wise performance

## API Integration

### Status Tracking

The system integrates with external APIs for status tracking:

- **Dataset Status**: Track analysis and processing status
- **Model Status**: Monitor training progress and completion
- **Authentication**: Support for both Bearer token and x-api-key methods

### Endpoints

- `GET /api/external/datasets/{id}/status` - Get dataset status
- `PUT /api/external/datasets/status` - Update dataset status
- `GET /api/external/models/{id}/status` - Get model status
- `PUT /api/external/models/status` - Update model status

## Sampling Strategies

### Files Per Writer

Calculate sampling based on episode requirements:
```
sampling_value = (n_shot + n_query) * multiplier
```

### Percentage Sampling

Use a percentage of total files:
```
sampling_value = percentage * total_files
```

## Performance Optimization

### GPU Acceleration

- Automatic CUDA detection and utilization
- Mixed precision training support
- Memory-efficient batch processing

### Data Loading

- Multi-process data loading
- Prefetching for improved throughput
- Memory-mapped file access

### Caching

- Embedding caching for faster evaluation
- Model state caching for checkpointing
- Dataset statistics caching

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Increase number of workers
   - Use SSD storage for datasets
   - Enable data prefetching

3. **API Connection Issues**
   - Verify API credentials
   - Check network connectivity
   - Validate endpoint URLs

### Debug Mode

Enable verbose logging by setting environment variables:
```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ProtoNet paper: "Prototypical Networks for Few-shot Learning"
- PyTorch team for the deep learning framework
- Azure team for cloud storage services
- Open source community for various dependencies 