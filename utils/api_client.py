import requests
import urllib3
from typing import Dict, Optional
import os


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ApiClient:
    
    def __init__(self, base_url: str = None, api_key: str = None):
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
        


    def get_dataset_status(self, task_id: str) -> Optional[Dict]:
        if not self.api_key:
            print("Skipping dataset status check - API_KEY not configured")
            return None
            
        try:
            url = f"{self.base_url}/api/external/datasets/{task_id}/status"
            response = requests.get(
                url, 
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request to get dataset status: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing API response: {e}")
            return None

    def update_dataset_status(self, dataset_id: str, status: int, message: str = "") -> bool:
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
            
            response = requests.put(
                url, 
                json=payload,
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating dataset status: {e}")
            return False

    def get_model_status(self, model_id: str) -> Optional[Dict]:
        if not self.api_key:
            print("Skipping model status check - API_KEY not configured")
            return None
            
        try:
            url = f"{self.base_url}/api/external/models/{model_id}/status"
            response = requests.get(
                url, 
                headers=self.headers,
                verify=False,  # Disable SSL verification for localhost
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request to get model status: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing model status API response: {e}")
            return None

    def update_model_status(self, model_id: str, status: int, message: str = "") -> bool:
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
            return self._try_bearer_auth(url, payload)
    
    def _try_x_api_key_auth(self, url: str, payload: dict, fallback_on_401: bool = False) -> bool:
        try:
            response = requests.put(
                url, 
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 401 and fallback_on_401:
                return False
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if "401" in str(e) and fallback_on_401:
                return False
            else:
                return False
        
        return True
    
    def _try_bearer_auth(self, url: str, payload: dict) -> bool:
        try:
            response = requests.put(
                url, 
                json=payload,
                headers=self.model_headers,
                verify=False,
                timeout=30
            )
            
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            return False