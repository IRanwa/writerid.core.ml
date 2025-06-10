# WriterID Core ML

A machine learning pipeline for writer identification using few-shot learning with prototypical networks.

## Project Structure

```
WriterID Core ML/
├── entry_points/          # Main entry points
│   ├── queue_listener.py  # Primary entry point - Azure Queue listener
│   └── main.py           # Testing entry point - Local testing
├── core/                 # Core ML processing modules
│   ├── executor.py       # Main task execution and training pipeline
│   ├── task_processor.py # Azure-based task processing
│   └── sampler.py        # Episodic sampling for few-shot learning
├── models/               # Neural network models and architectures
│   ├── models.py         # Prototypical network implementation
│   └── network_architectures.py # Backbone network handlers
├── data/                 # Data processing and management
│   ├── data_manager.py   # Dataset loading and management
│   └── image_processor.py # Image preprocessing utilities
├── utils/                # Utility functions
│   ├── training_utils.py # Training and evaluation utilities
│   ├── plotting_utils.py # Visualization and plotting functions
│   └── api_client.py     # External API communication client
├── temp_data/            # Temporary data storage
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── .dockerignore        # Docker ignore patterns
```

## Entry Points

### Primary Entry Point: `queue_listener.py`
- **Purpose**: Production entry point that listens to Azure Storage Queue for tasks
- **Usage**: Processes incoming tasks for dataset analysis and model training
- **Location**: `entry_points/queue_listener.py`

### Testing Entry Point: `main.py`
- **Purpose**: Local testing and development
- **Usage**: Runs experiments directly without queue system
- **Location**: `entry_points/main.py`

## Running the Application

### Production Mode (Queue Listener)
```bash
python entry_points/queue_listener.py
```

### Testing Mode (Direct Execution)
```bash
python entry_points/main.py
```



## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:
```
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
QUEUE_NAME=your_queue_name
API_BASE_URL=https://localhost:44302
API_KEY=WID-API-2024-SecureKey-XYZ789
```

## API Integration

The system now integrates with an external API for dataset management:

### Queue Message Formats

**Dataset Analysis:**
```json
{
  "task": "analyze_dataset",
  "taskId": "5b5e2e87-e146-4c3b-84ce-35c035f7d639",
  "container_name": "dataset-5b5e2e87-e146-4c3b-84ce-35c035f7d639"
}
```

**Model Training:**
```json
{
  "task": "train",
  "dataset_container_name": "dataset-container-name",
  "model_container_name": "model-container-name"
}
```

### API Endpoints Used
- `GET /api/external/datasets/{id}/status` - Get dataset information
- `POST /api/external/datasets/status` - Update processing status

### API Authentication
All API requests include the following header:
```
x-api-key: WID-API-2024-SecureKey-XYZ789
```

### API Client
The system uses a dedicated `ApiClient` class located in `utils/api_client.py` that handles:
- Authentication headers
- SSL certificate handling for localhost
- Error handling and retries
- Request/response logging

### Processing Status Values
- `0` - Created
- `1` - Processing (required to start)
- `2` - Completed (success)
- `3` - Failed (error occurred)



## Docker Support

Build and run with Docker:
```bash
docker build -t writerid-core-ml .
docker run writerid-core-ml
``` 