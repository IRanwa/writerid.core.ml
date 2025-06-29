import sys
import json
import os
import time
import traceback
import base64
import binascii
from dotenv import load_dotenv


try:
    from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
except ImportError as e:
    print(f"Failed to import Azure Storage Queue library: {e}")
    sys.exit(1)


try:
    from core.task_processor import TaskProcessor
except ImportError as e:
    print(f"Failed to import TaskProcessor: {e}")
    sys.exit(1)

class QueueMessageListener:
    def __init__(self, connection_string, queue_name):
        self.connection_string = connection_string
        self.queue_name = queue_name
        self.queue_client = None
        self.task_processor = None
        self._initialize_clients()
    
    def _decode_message_content(self, raw_content):
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            try:
                decoded_bytes = base64.b64decode(raw_content)
                decoded_string = decoded_bytes.decode('utf-8')
                return json.loads(decoded_string)
            except (base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"Failed to decode message content")
                raise json.JSONDecodeError(f"Could not decode message: {e}", raw_content, 0)
        
    def _initialize_clients(self):
        try:
            self.queue_client = QueueClient.from_connection_string(
                self.connection_string,
                self.queue_name
            )
            self.task_processor = TaskProcessor(self.connection_string)
        except Exception as e:
            print(f"Failed to initialize clients: {e}")
            raise

    def _reconnect(self):
        try:
            self._initialize_clients()
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False

    def _clear_corrupted_messages(self):
        print("Clearing corrupted messages...")
        try:
            corrupted_count = 0
            max_attempts = 10
            
            for attempt in range(max_attempts):
                try:
                    response = self.queue_client._client.messages.dequeue(
                        number_of_messages=1,
                        visibility_timeout=30
                    )
                    
                    if not hasattr(response, 'queue_message') or not response.queue_message:
                        break
                        
                    message = response.queue_message[0] if response.queue_message else None
                    if message:
                        try:
                            self.queue_client._client.messages.delete(
                                message.message_id,
                                message.pop_receipt
                            )
                            corrupted_count += 1
                        except Exception as delete_error:
                            print(f"Failed to delete corrupted message: {delete_error}")
                            break
                    else:
                        break
                        
                except Exception as process_error:
                    print(f"Error processing message during cleanup: {process_error}")
                    break
                    
            if corrupted_count > 0:
                print(f"Removed {corrupted_count} corrupted messages")
            return corrupted_count > 0
            
        except Exception as e:
            print(f"Failed to clear corrupted messages: {e}")
            return False

    def handle_analyze_dataset(self, message_data):
        try:
            task_id = message_data.get("taskId")
            container_name = message_data.get("container_name")
            
            if not task_id:
                print("Invalid message format: missing taskId")
                return
                
            if not container_name:
                print("Invalid message format: missing container_name")
                return
            
            print(f"Processing dataset analysis: {task_id}")
            result = self.task_processor.analyze_dataset(task_id, container_name)
            
            if result:
                print("Dataset analysis completed successfully")
            else:
                print("Dataset analysis failed")
                
        except Exception as e:
            print(f"Error in dataset analysis: {str(e)}")
            traceback.print_exc()

    def listen(self):
        print(f"Listening for messages on queue: {self.queue_client.queue_name}")
        while True:
            try:
                messages = []
                try:
                    messages = list(self.queue_client.receive_messages(
                        messages_per_page=1, 
                        visibility_timeout=30
                    ))
                except Exception as receive_error:
                    error_msg = str(receive_error).lower()
                    if 'decode' in error_msg or 'utf-8' in error_msg or 'pipelineresponse' in error_msg:
                        print("Message decoding error - clearing corrupted messages...")
                        if self._clear_corrupted_messages():
                            continue
                        else:
                            time.sleep(10)
                            continue
                    else:
                        raise receive_error
                
                if not messages:
                    time.sleep(5)
                    continue
                    
                for message in messages:
                    message_processed = False
                    
                    try:
                        message_content = self._decode_message_content(message.content)
                        task = message_content.get("task")

                        if task == "analyze_dataset":
                            self.handle_analyze_dataset(message_content)
                            message_processed = True

                        elif task == "train":
                            dataset_container_name = message_content.get("dataset_container_name")
                            model_container_name = message_content.get("model_container_name")
                            if dataset_container_name and model_container_name:
                                print(f"Starting model training: {dataset_container_name}")
                                result = self.task_processor.train_model(dataset_container_name, model_container_name)
                                print("Model training completed")
                                message_processed = True
                            else:
                                print("Missing container names in train message")

                        else:
                            print(f"Unknown task: {task}")

                        if message_processed:
                            try:
                                self.queue_client.delete_message(message)
                            except Exception as delete_error:
                                print(f"Failed to delete message: {delete_error}")

                    except json.JSONDecodeError as json_error:
                        print(f"Malformed message - deleting")
                        try:
                            self.queue_client.delete_message(message)
                        except Exception as delete_error:
                            print(f"Failed to delete malformed message: {delete_error}")
                            
                    except Exception as processing_error:
                        print(f"Error processing message: {processing_error}")
                        traceback.print_exc()
                
                time.sleep(5)

            except Exception as e:
                print(f"Error in message listener: {e}")
                
                error_message = str(e).lower()
                
                if any(keyword in error_message for keyword in ['decode', 'utf-8', 'unicodedecodeerror']):
                    print("Message decoding error - clearing corrupted messages...")
                    if self._clear_corrupted_messages():
                        continue
                        
                elif any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'pipelineresponse']):
                    print("Connection error - attempting to reconnect...")
                    if self._reconnect():
                        continue
                
                time.sleep(10)

if __name__ == "__main__":
    print("Starting WriterID Core ML Queue Listener...")
    try:
        load_dotenv()

        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        queue_name = os.environ.get("QUEUE_NAME")
        api_base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")

        missing_vars = []
        if not connection_string:
            missing_vars.append("AZURE_STORAGE_CONNECTION_STRING")
        if not queue_name:
            missing_vars.append("QUEUE_NAME")
        if not api_base_url:
            missing_vars.append("API_BASE_URL")
        if not api_key:
            missing_vars.append("API_KEY")

        if missing_vars:
            print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
            print("Required .env file content:")
            print("AZURE_STORAGE_CONNECTION_STRING=your_connection_string")
            print("QUEUE_NAME=your_queue_name")
            print("API_BASE_URL=https://localhost:44302")
            print("API_KEY=WID-API-2024-SecureKey-XYZ789")
        else:
            listener = QueueMessageListener(connection_string, queue_name)
            listener.listen()
    except Exception as e:
        print(f"Fatal error: {e}")