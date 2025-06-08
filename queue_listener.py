import sys
print("Python interpreter:", sys.executable)
print("Script execution started.")
import json
import os
import time
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
from task_processor import TaskProcessor
from dotenv import load_dotenv

class QueueMessageListener:
    def __init__(self, connection_string, queue_name):
        self.queue_client = QueueClient.from_connection_string(
            connection_string,
            queue_name,
            message_encode_policy=TextBase64EncodePolicy(),
            message_decode_policy=TextBase64DecodePolicy()
        )
        self.task_processor = TaskProcessor(connection_string)

    def listen(self):
        print(f"Listening for messages on queue: {self.queue_client.queue_name}")
        while True:
            try:
                messages = self.queue_client.receive_messages(messages_per_page=1, visibility_timeout=30)
                for message in messages:
                    print(f"Received message: {message.content}")
                    try:
                        message_content = json.loads(message.content)
                        task = message_content.get("task")
                        params = message_content.get("params")

                        if task == "analyze_dataset":
                            container_name = params.get("container_name")
                            if container_name:
                                print(f"Starting dataset analysis for container '{container_name}'")
                                result = self.task_processor.analyze_dataset(container_name)
                                print(f"Analysis result: {result}")
                            else:
                                print("Missing 'container_name' in params.")

                        elif task == "train":
                            dataset_container_name = params.get("dataset_container_name")
                            model_container_name = params.get("model_container_name")
                            if dataset_container_name and model_container_name:
                                print(f"Starting model training for dataset '{dataset_container_name}'")
                                result = self.task_processor.train_model(dataset_container_name, model_container_name)
                                print(f"Training result: {result}")
                            else:
                                print("Missing 'dataset_container_name' or 'model_container_name' in params.")

                        else:
                            print(f"Unknown task: {task}")

                        self.queue_client.delete_message(message)
                        print(f"Message processed and deleted.")

                    except json.JSONDecodeError:
                        print(f"Failed to decode message content: {message.content}")
                    except Exception as e:
                        print(f"An error occurred while processing message: {e}")
                
                time.sleep(5)

            except Exception as e:
                print(f"An error occurred while listening for messages: {e}")
                time.sleep(10)

if __name__ == "__main__":
    print("Starting queue listener...")
    try:
        if load_dotenv():
            print(".env file found and loaded.")
        else:
            print(".env file not found.")

        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        queue_name = os.environ.get("QUEUE_NAME")

        if not connection_string or not queue_name:
            print("ERROR: Could not find AZURE_STORAGE_CONNECTION_STRING or QUEUE_NAME in environment variables.")
            print("Please ensure your .env file is present, saved with UTF-8 encoding (without BOM), and contains the correct variable names.")
            print("If you don't have a .env file, please create one with the following content:")
            print("AZURE_STORAGE_CONNECTION_STRING=your_connection_string")
            print("QUEUE_NAME=your_queue_name")

        else:
            print("Environment variables loaded. Initializing listener...")
            listener = QueueMessageListener(connection_string, queue_name)
            listener.listen()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")