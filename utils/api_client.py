import requests
import urllib3
from typing import Dict, Optional
import os


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ApiClient:
    """
    API client for communicating with the external dataset management API.
    """
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API (defaults to env variable)
            api_key: API key for authentication (defaults to env variable)
        """
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://localhost:44302")
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.model_auth_method = os.getenv("MODEL_API_AUTH_METHOD", "auto").lower()
        
        if not self.api_key:
            print("WARNING: API_KEY not configured. Model and dataset status tracking will not work.")
            print("Please set API_KEY environment variable or in config.env file.")
            # Don't raise error to allow system to work without API tracking
        
        self.base_url = self.base_url.rstrip('/')
        
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        # Alternative headers for model API (in case it uses different authentication)
        self.model_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"API Client initialized with base URL: {self.base_url}")
        print(f"API Key configured: {'Yes' if self.api_key else 'No'} (length: {len(self.api_key) if self.api_key else 0})")
        print(f"Model API auth method: {self.model_auth_method}")

    def get_dataset_status(self, task_id: str) -> Optional[Dict]:
        """
        Get dataset status from API.
        
        Args:
            task_id: The task/dataset ID to check
            
        Returns:
            Dict: Dataset information or None if failed
        """
        if not self.api_key:
            print("Skipping dataset status check - API_KEY not configured")
            return None
            
        try:
            url = f"{self.base_url}/api/external/datasets/{task_id}/status"
            print(f"Making API GET request to: {url}")
            
            response = requests.get(
                url, 
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"API Response: {data}")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request to get dataset status: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing API response: {e}")
            return None

    def update_dataset_status(self, dataset_id: str, status: int, message: str = "") -> bool:
        """
        Update dataset status via API.
        
        Args:
            dataset_id: The dataset ID to update
            status: Status code (0=Created, 1=Processing, 2=Completed, 3=Failed)
            message: Optional status message
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.api_key:
            print("Skipping dataset status update - API_KEY not configured")
            return False
            
        try:
            url = f"{self.base_url}/api/external/datasets/status"
            payload = {
                "datasetId": dataset_id,
                "status": status,
                "message": message
            }
            
            print(f"Making API PUT request to: {url}")
            print(f"Payload: {payload}")
            
            response = requests.put(
                url, 
                json=payload,
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            
            print(f"Status update successful: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating dataset status: {e}")
            return False

    def get_model_status(self, model_id: str) -> Optional[Dict]:
        """
        Get model status from API.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            Dict: Model status information or None if failed
            Format: {"status": str, "message": str, ...}
            Note: Status is returned as string (e.g., "Created", "Processing")
        """
        if not self.api_key:
            print("Skipping model status check - API_KEY not configured")
            return None
            
        try:
            url = f"{self.base_url}/api/external/models/{model_id}/status"
            print(f"Making API GET request to: {url}")
            
            response = requests.get(
                url, 
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"Model status API Response: {data}")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request to get model status: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing model status API response: {e}")
            return None

    def update_model_status(self, model_id: str, status: int, message: str = "") -> bool:
        """
        Update model status via API.
        
        Args:
            model_id: The model ID to update
            status: Status code (0=Created, 1=Processing, 2=Completed, 3=Failed, 4=Reconfigure)
            message: Optional status message
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = f"{self.base_url}/api/external/models/status"
        payload = {
            "ModelId": model_id,
            "Status": status
        }
        
        if message:
            payload["Message"] = message
        
        # Check if API key is configured
        if not self.api_key:
            print("Skipping model status update - API_KEY not configured")
            return False
        
        print(f"Making API PUT request to: {url}")
        print(f"Payload: {payload}")
        print(f"Using authentication method: {self.model_auth_method}")
        
        # Use configured authentication method
        if self.model_auth_method == "bearer":
            return self._try_bearer_auth(url, payload)
        elif self.model_auth_method == "x-api-key":
            return self._try_x_api_key_auth(url, payload)
        else:  # auto - try both methods
            # Try x-api-key authentication first (same as dataset API)
            if self._try_x_api_key_auth(url, payload, fallback_on_401=True):
                return True
            # If x-api-key failed with 401, try Bearer token
            print(f"Falling back to Bearer token authentication...")
            return self._try_bearer_auth(url, payload)
    
    def _try_x_api_key_auth(self, url: str, payload: dict, fallback_on_401: bool = False) -> bool:
        """Try authentication with x-api-key header."""
        try:
            print(f"Trying x-api-key authentication...")
            print(f"Headers: {dict(self.headers)}")
            
            response = requests.put(
                url, 
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=30
            )
            
            print(f"Response status code: {response.status_code}")
            if response.status_code == 200:
                print(f"Model status update successful with x-api-key: {response.status_code}")
                return True
            elif response.status_code == 401 and fallback_on_401:
                print(f"x-api-key authentication failed (401)")
                print(f"Response content: {response.text}")
                return False
            else:
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response content: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if "401" in str(e) and fallback_on_401:
                print(f"x-api-key authentication failed: {e}")
                return False
            else:
                print(f"Error with x-api-key authentication: {e}")
                return False
        
        return True
    
    def _try_bearer_auth(self, url: str, payload: dict) -> bool:
        """Try authentication with Bearer token."""
        try:
            print(f"Trying Bearer token authentication...")
            print(f"Headers: {dict(self.model_headers)}")
            
            response = requests.put(
                url, 
                json=payload,
                headers=self.model_headers,
                verify=False,
                timeout=30
            )
            
            print(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response content: {response.text}")
            
            response.raise_for_status()
            
            print(f"Model status update successful with Bearer token: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating model status with Bearer token: {e}")
            return False