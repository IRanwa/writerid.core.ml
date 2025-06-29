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
        
        if not self.api_key:
            raise ValueError("API_KEY must be provided either as parameter or environment variable")
            
        self.base_url = self.base_url.rstrip('/')
        
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        print(f"API Client initialized with base URL: {self.base_url}")

    def get_dataset_status(self, task_id: str) -> Optional[Dict]:
        """
        Get dataset status from API.
        
        Args:
            task_id: The task/dataset ID to check
            
        Returns:
            Dict: Dataset information or None if failed
        """
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