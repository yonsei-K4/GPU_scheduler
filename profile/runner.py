import torch
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, Any, List
from threading import Thread
from queue import Queue
import azure.functions as func
import logging
import requests

class ModelRunner:
    model_list = []
    runtime_info = {}
    sessions = {}
    def __init__(self, model_list: List[str], runtime_info: Dict[str, Dict[int, int]]):
        ModelRunner.model_list = model_list
        ModelRunner.runtime_info = runtime_info
        ModelRunner.sessions = {}
        ModelRunner._setup_sessions()

    @staticmethod
    def _setup_sessions():
        gpu_count = torch.cuda.device_count()
        for gpu_id_str, model_configs in ModelRunner.runtime_info.items():
            gpu_id = int(gpu_id_str)
            if gpu_id >= gpu_count:
                raise ValueError(f"GPU ID {gpu_id} exceeds available GPUs ({gpu_count})")
            
            for model_idx in model_configs.keys():
                if not isinstance(model_idx, int) or model_idx < 0 or model_idx >= len(ModelRunner.model_list):
                    raise ValueError(f"Invalid model index {model_idx}")
                
                model_path = ModelRunner.model_list[model_idx]
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("CUDAExecutionProvider not available")
                
                session_options = ort.SessionOptions()
                print(f"Setting up model {model_idx} ({model_path}) on GPU {gpu_id}")
                session = ort.InferenceSession(
                    model_path,
                    providers=[("CUDAExecutionProvider", {"device_id": gpu_id})],
                    sess_options=session_options
                )
                ModelRunner.sessions[(gpu_id, model_idx)] = session
                print(f"Loaded model {model_idx} on GPU {gpu_id}")

    @staticmethod
    def _run_model_thread(gpu_id: int, model_idx: int, batch_size: int, result_queue: Queue):
        """Helper function to run a model in a thread."""
        session = ModelRunner.sessions[(gpu_id, model_idx)]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        height, width = (640, 640) if "yolo" in ModelRunner.model_list[model_idx].lower() else (224, 224)
        input_data = np.random.randn(batch_size, 3, height, width).astype(np.float32)
        
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_data})
        exec_time = time.time() - start_time
        
        output_tensor = torch.from_numpy(outputs[0])
        
        runtime_details = {
            "gpu_id": gpu_id,
            "model_idx": model_idx,
            "model_path": ModelRunner.model_list[model_idx],
            "batch_size": batch_size,
            "input_shape": input_data.shape,
            "output_shape": output_tensor.shape,
            "execution_time_s": exec_time
        }
        result_queue.put(((gpu_id, model_idx), runtime_details))

    @staticmethod
    def run_all() -> Dict[tuple, Dict[str, Any]]:
        result_queue = Queue()
        threads = []
        
        for gpu_id_str, model_configs in ModelRunner.runtime_info.items():
            gpu_id = int(gpu_id_str)
            for model_idx, batch_size in model_configs.items():
                t = Thread(target=ModelRunner._run_model_thread, args=(gpu_id, model_idx, batch_size, result_queue))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        results = {}
        while not result_queue.empty():
            key, result = result_queue.get()
            results[key] = result
        
        for (gpu_id, model_idx), result in results.items():
            print(f"Ran model {model_idx} ({ModelRunner.model_list[model_idx]}) on GPU {gpu_id}: {result['execution_time_s']:.3f}s")
        
        return results

if __name__ == "__main__":
    model_list = [
        "models/alexnet_dynamic.onnx",
        "models/vgg19_dynamic.onnx",
        "models/resnet18_dynamic.onnx",
        "models/resnet50_dynamic.onnx"
    ]
    
    runtime_info = {
        "0": {3: 4, 0: 24},
        "1": {0: 2, 1: 20},
        "2": {1: 1},
        "3": {2: 1}
    }
    
    runner = ModelRunner(model_list, runtime_info)
    results = runner.run_all()
    