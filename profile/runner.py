import torch
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, Any, List
from threading import Thread
from queue import Queue

class ModelRunner:
    def __init__(self, model_list: List[str], runtime_info: Dict[str, Dict[int, int]]):
        self.model_list = model_list
        self.runtime_info = runtime_info
        self.sessions = {}
        self._setup_sessions()

    def _setup_sessions(self):
        gpu_count = torch.cuda.device_count()
        for gpu_id_str, model_configs in self.runtime_info.items():
            gpu_id = int(gpu_id_str)
            if gpu_id >= gpu_count:
                raise ValueError(f"GPU ID {gpu_id} exceeds available GPUs ({gpu_count})")
            
            for model_idx in model_configs.keys():
                if not isinstance(model_idx, int) or model_idx < 0 or model_idx >= len(self.model_list):
                    raise ValueError(f"Invalid model index {model_idx}")
                
                model_path = self.model_list[model_idx]
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("CUDAExecutionProvider not available")
                
                session_options = ort.SessionOptions()
                print(f"Setting up model {model_idx} ({model_path}) on GPU {gpu_id}")
                session = ort.InferenceSession(
                    model_path,
                    providers=[("CUDAExecutionProvider", {"device_id": gpu_id})],
                    sess_options=session_options
                )
                self.sessions[(gpu_id, model_idx)] = session
                print(f"Loaded model {model_idx} on GPU {gpu_id}")

    def _run_model_thread(self, gpu_id: int, model_idx: int, batch_size: int, result_queue: Queue):
        """Helper function to run a model in a thread."""
        session = self.sessions[(gpu_id, model_idx)]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        height, width = (640, 640) if "yolo" in self.model_list[model_idx].lower() else (224, 224)
        input_data = np.random.randn(batch_size, 3, height, width).astype(np.float32)
        
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_data})
        exec_time = time.time() - start_time
        
        output_tensor = torch.from_numpy(outputs[0])
        
        runtime_details = {
            "gpu_id": gpu_id,
            "model_idx": model_idx,
            "model_path": self.model_list[model_idx],
            "batch_size": batch_size,
            "input_shape": input_data.shape,
            "output_shape": output_tensor.shape,
            "execution_time_s": exec_time
        }
        result_queue.put(((gpu_id, model_idx), runtime_details))

    def run_model(self, gpu_id: int, model_idx: int, batch_size: int = None) -> Dict[str, Any]:
        if (gpu_id, model_idx) not in self.sessions:
            raise ValueError(f"No session for GPU {gpu_id} and model {model_idx}")
        
        batch_size = batch_size if batch_size is not None else self.runtime_info[str(gpu_id)][model_idx]
        result_queue = Queue()
        
        t = Thread(target=self._run_model_thread, args=(gpu_id, model_idx, batch_size, result_queue))
        t.start()
        t.join()
        
        key, result = result_queue.get()
        return result

    def run_all(self) -> Dict[tuple, Dict[str, Any]]:
        result_queue = Queue()
        threads = []
        
        for gpu_id_str, model_configs in self.runtime_info.items():
            gpu_id = int(gpu_id_str)
            for model_idx, batch_size in model_configs.items():
                t = Thread(target=self._run_model_thread, args=(gpu_id, model_idx, batch_size, result_queue))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        results = {}
        while not result_queue.empty():
            key, result = result_queue.get()
            results[key] = result
        
        for (gpu_id, model_idx), result in results.items():
            print(f"Ran model {model_idx} ({self.model_list[model_idx]}) on GPU {gpu_id}: {result['execution_time_s']:.3f}s")
        
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
    
    # custom_result = runner.run_model(gpu_id=0, model_idx=3, batch_size=6)
    # print(f"\nCustom run: {custom_result}")