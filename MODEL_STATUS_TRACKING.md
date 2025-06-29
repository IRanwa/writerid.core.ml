# Model Status Tracking

## Overview
The system now automatically tracks model training status through API calls to the portal backend. When a training job starts from a queue message, the system extracts the model ID from the container name and updates the model status throughout the training process.

## Status Values
Based on the `ProcessingStatus` enum in the portal:

```csharp
public enum ProcessingStatus
{
    Created = 0,      // The operation has been created but not yet started
    Processing = 1,   // The operation is currently being processed
    Completed = 2,    // The operation has been successfully completed
    Failed = 3,       // The operation has failed during processing
    Reconfigure = 4   // The operation needs reconfiguration and can be restarted
}
```

**Status Values**: 
- **GET requests return**: String values ("Created", "Processing", "Completed", "Failed", "Reconfigure")
- **PUT requests send**: Numeric codes (0, 1, 2, 3, 4 corresponding to enum values above)
- **System handles both**: Code automatically handles both string and numeric status formats

## How It Works

### 1. Model ID Extraction
- **Container Name Format**: `model-{id}` (e.g., `model-1`, `model-123`, `model-70fb5fb0-9b8a-4b9f-9f6f-2cc4e7b11de8`)
- **Extraction Logic**: Removes the `model-` prefix to get the ID
- **Examples**: 
  - `model-1` → `1`
  - `model-123` → `123`
  - `model-70fb5fb0-9b8a-4b9f-9f6f-2cc4e7b11de8` → `70fb5fb0-9b8a-4b9f-9f6f-2cc4e7b11de8`

### 2. API Endpoints

#### Update Model Status (PUT)
- **URL**: `PUT {API_BASE_URL}/api/external/models/status`
- **Headers**: Content-Type: application/json, x-api-key: {API_KEY}
- **Body**:
  ```json
  {
    "ModelId": "123",
    "Status": 1,
    "Message": "Optional status message"
  }
  ```

#### Get Model Status (GET)
- **URL**: `GET {API_BASE_URL}/api/external/models/{id}/status`
- **Headers**: x-api-key: {API_KEY}
- **Response**:
  ```json
  {
    "status": "Created",
    "message": "Model created and ready for training"
  }
  ```
  
  **Note**: Status can be returned as either string values ("Created", "Processing", etc.) or numeric codes (0, 1, etc.)

### 3. Status Update Flow

#### Pre-Training Validation
- **Action**: GET request to check current model status
- **Requirement**: Model must be in `Created (0)` or `Reconfigure (4)` status
- **Behavior**: Training is aborted if model is in `Processing (1)`, `Completed (2)`, or `Failed (3)` status

#### Training Start
- **Status**: `Processing (1)`
- **Message**: "Model training started"
- **Triggered**: When `train_model()` method is called and status validation passes

#### Training Success
- **Status**: `Completed (2)`
- **Message**: "Model training completed successfully"
- **Triggered**: When training completes successfully and model is uploaded

#### Training Failure
- **Status**: `Failed (3)`
- **Message**: Specific error message describing the failure
- **Triggered**: When any error occurs during:
  - Dataset download failure
  - Training execution errors
  - Model upload failures
  - Any other exceptions

#### Reconfigure Status
- **Status**: `Reconfigure (4)`
- **Usage**: Set by portal when model needs to be retrained with new configuration
- **Behavior**: Allows training to proceed (same as `Created` status)

## Implementation Details

### Files Modified

#### 1. `utils/api_client.py`
- Added `get_model_status()` method for GET requests to check current status
- Added `update_model_status()` method for PUT requests to update status
- Updated API endpoint structure to use ModelId in request body
- Handles HTTP PUT request to model status endpoint
- Includes error handling and logging

#### 2. `core/task_processor.py`
- Added `_extract_model_id()` method for ID extraction
- Added `_get_model_status()` wrapper method for status checking
- Added `_update_model_status()` wrapper method for status updates
- Added `_can_start_training()` method for pre-training validation
- Enhanced status handling to support both string and numeric status values
- Integrated status validation and updates into `train_model()` method:
  - Model ID extraction validation (training blocked if extraction fails)
  - Status validation before training starts (training blocked if not Created/Reconfigure)
  - Status check failure handling (training blocked if status cannot be retrieved)
  - Flexible status format handling (accepts both "Created"/0 and "Reconfigure"/4)
  - Processing status at start (only if validation passes)
  - Completed status on success
  - Failed status on any error

#### 3. `entry_points/queue_listener.py`
- Added documentation comments explaining the automatic status tracking
- Enhanced logging to show container name mapping

## Usage Examples

### Successful Training Flow
```
Queue Message: { "task": "train", "model_container_name": "model-123", ... }
↓
Extract model ID: "123"
↓
Check current status: GET /api/external/models/123/status (must be Created or Reconfigure)
↓
API Call: PUT /api/external/models/status { "ModelId": "123", "Status": 1, "Message": "Model training started" }
↓
Training executes successfully
↓
API Call: PUT /api/external/models/status { "ModelId": "123", "Status": 2, "Message": "Model training completed successfully" }
```

### Failed Training Flow
```
Queue Message: { "task": "train", "model_container_name": "model-456", ... }
↓
Extract model ID: "456"
↓
Check current status: GET /api/external/models/456/status (must be Created or Reconfigure)
↓
API Call: PUT /api/external/models/status { "ModelId": "456", "Status": 1, "Message": "Model training started" }
↓
Error occurs (e.g., dataset download fails)
↓
API Call: PUT /api/external/models/status { "ModelId": "456", "Status": 3, "Message": "Failed to download dataset from container: dataset-xyz" }
```

## Error Handling

### Invalid Container Name Format
- If container name doesn't follow `model-{id}` format
- Warning logged but training continues
- No status updates performed

### API Call Failures
- If status update API calls fail, errors are logged
- Training process continues regardless
- System is resilient to API availability issues

### Network Issues
- API calls have 30-second timeout
- SSL verification disabled for localhost development
- Detailed error logging for troubleshooting

## Configuration

### Environment Variables
The model status tracking uses the same API configuration as dataset status tracking:

```bash
API_BASE_URL=https://localhost:44302      # Base URL for API calls
API_KEY=your_api_key_here                # Authentication key (REQUIRED)
MODEL_API_AUTH_METHOD=auto               # Authentication method: auto, x-api-key, bearer
```

**Important**: The `API_KEY` must be configured for status tracking to work. If not configured, the system will log warnings but continue to function without status updates.

#### Authentication Methods:
- **auto** (default): Try x-api-key first, fallback to Bearer token if 401 error
- **x-api-key**: Use `x-api-key: {token}` header (same as dataset API)  
- **bearer**: Use `Authorization: Bearer {token}` header

### Container Naming Convention
Ensure model container names follow the pattern:
- ✅ `model-1`
- ✅ `model-123`
- ✅ `model-abc123`
- ✅ `model-70fb5fb0-9b8a-4b9f-9f6f-2cc4e7b11de8` (GUID)
- ❌ `model123` (missing dash)
- ❌ `training-1` (wrong prefix)

## Logging Output

### Successful Status Update
```
Extracted model ID '123' from container name 'model-123'
Making API PUT request to: https://localhost:44302/api/external/models/status
Payload: {'ModelId': '123', 'Status': 1, 'Message': 'Model training started'}
Model status update successful: 200
Model 123 status updated to Processing
```

### Failed Status Update
```
Extracted model ID '456' from container name 'model-456'
Making API PUT request to: https://localhost:44302/api/external/models/status
Error updating model status: HTTPSConnectionPool(host='localhost', port=44302): Max retries exceeded
Failed to update model 456 status to Processing
```

## Status Validation

### Training Prerequisites
Before starting training, the system now validates the current model status:

1. **GET Request**: Checks current model status using `GET /api/external/models/{id}/status`
2. **Valid Statuses**: Training only proceeds if model is in:
   - `"Created"` or `0`: Initial state, ready for first training
   - `"Reconfigure"` or `4`: Model marked for retraining with new configuration
3. **Invalid Statuses**: Training is aborted if model is in:
   - `"Processing"` or `1`: Already being trained
   - `"Completed"` or `2`: Training already completed
   - `"Failed"` or `3`: Previous training failed (should be reset to Created/Reconfigure first)
4. **Status Check Failure**: Training is aborted if status cannot be retrieved (for safety)

### Example Status Check Flow
```
Current Status: "Created" (or 0) → Training Allowed ✅
Current Status: "Reconfigure" (or 4) → Training Allowed ✅
Current Status: "Processing" (or 1) → Training Blocked ❌
Current Status: "Completed" (or 2) → Training Blocked ❌
Current Status: "Failed" (or 3) → Training Blocked ❌
Status Check Failed → Training Blocked ❌
Invalid Container Name → Training Blocked ❌
```

## Benefits

1. **Real-time Status Tracking**: Portal users can see training progress in real-time
2. **Automatic Updates**: No manual intervention required
3. **Error Transparency**: Failed jobs are automatically marked with error details
4. **Resilient Design**: Training continues even if status updates fail
5. **Detailed Logging**: Full audit trail of status changes
6. **Consistent API**: Uses same patterns as dataset status tracking
7. **Status Validation**: Prevents duplicate or invalid training attempts
8. **Reconfigure Support**: Allows models to be retrained when needed
9. **Fail-Safe Design**: Blocks training if status cannot be verified or container name is invalid

## Troubleshooting

### Common Issues

#### 1. 401 Unauthorized Error
**Problem**: `Error updating model status: 401 Client Error: Unauthorized`

**Solutions**:
1. **Check API Key**: Ensure `API_KEY` is configured in `config.env` or environment variables
2. **Try Different Auth Method**: Set `MODEL_API_AUTH_METHOD=bearer` or `MODEL_API_AUTH_METHOD=x-api-key`
3. **Verify API Key**: Check that the API key is valid and has correct permissions
4. **Check Endpoint**: Confirm the model API endpoint requires different authentication than dataset API

#### 2. API Key Not Configured
**Problem**: `WARNING: API_KEY not configured`

**Solution**: 
1. Add `API_KEY=your_actual_api_key` to `config.env` file
2. Or set environment variable: `export API_KEY=your_actual_api_key`

#### 3. Authentication Method Issues
**Problem**: API calls work for dataset but fail for models

**Solution**: Set specific authentication method:
```bash
# If dataset API works with x-api-key but model API needs Bearer token
MODEL_API_AUTH_METHOD=bearer
```

### Debugging Steps

1. **Enable Detailed Logging**: Check console output for detailed API call information
2. **Verify Configuration**: Look for "API Client initialized" messages
3. **Check Authentication**: Watch for "Trying x-api-key authentication..." vs "Trying Bearer token authentication..."
4. **Examine Response**: Check response status codes and error messages

### Example Debug Output

#### Successful Authentication:
```
API Client initialized with base URL: https://localhost:44302
API Key configured: Yes (length: 32)
Model API auth method: auto
Extracted model ID '123' from container name 'model-123'
Making API PUT request to: https://localhost:44302/api/external/models/status
Using authentication method: auto
Trying x-api-key authentication...
Response status code: 200
Model status update successful with x-api-key: 200
Model 123 status updated to Processing
```

#### Failed Authentication with Fallback:
```
Trying x-api-key authentication...
Response status code: 401
x-api-key authentication failed (401)
Falling back to Bearer token authentication...
Trying Bearer token authentication...
Response status code: 200
Model status update successful with Bearer token: 200
Model 123 status updated to Processing
```

## Testing

### Manual Testing
1. Send training message with `model_container_name: "model-1"`
2. Check logs for model ID extraction and API calls
3. Verify status updates in portal backend
4. Test both success and failure scenarios

### Authentication Testing
1. Test with `MODEL_API_AUTH_METHOD=auto`
2. Test with `MODEL_API_AUTH_METHOD=x-api-key`
3. Test with `MODEL_API_AUTH_METHOD=bearer`
4. Test with missing API key
5. Test with invalid API key

### Common Test Scenarios
- Valid container name format
- Invalid container name format
- API endpoint unavailable
- Authentication failures
- Network timeouts
- Training success
- Training failures (dataset, execution, upload) 