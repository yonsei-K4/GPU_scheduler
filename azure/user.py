# user request generator( using curl command )
# send multiple request to the azure function app at once using threads
import subprocess
import threading

# List of curl commands
loader = "curl http://localhost:7071/api/loader"
listModels = "curl http://localhost:7071/api/list"
runModels = [
    "curl http://localhost:7071/api/runner?model=alexnet",
    "curl http://localhost:7071/api/runner?model=resnet18",
    "curl http://localhost:7071/api/runner?model=resnet50",
    "curl http://localhost:7071/api/runner?model=resnext101",
]

# Function to execute a curl command
def execute_curl(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command: {command}\nOutput: {result.stdout}\nError: {result.stderr}")
    except Exception as e:
        print(f"Error executing command {command}: {e}")

# Create and start threads for each curl command
execute_curl(loader)
execute_curl(listModels)
threads = []
for command in runModels:
    thread = threading.Thread(target=execute_curl, args=(command,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All curl commands executed.")