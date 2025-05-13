# Global Session Manager
import os
import torch
import numpy as np

class SessionManager:
    sessList = {}
    inputList = {}
    batchList = {}
    def __init__():
        pass
    
    def fetch_model(model_name: str, providers: tuple, batch_size: int = 1):
        if model_name not in SessionManager.sessList.keys():
            model_path = os.path.join(os.path.dirname(__file__), f"../profile/{model_name}.onnx")
            SessionManager.batchList[model_name] = batch_size
            
            import onnxruntime
            session = onnxruntime.InferenceSession(
                model_path,
                providers=[
                    providers,
                    # "TensorrtExecutionProvider",
                    # "CUDAExecutionProvider",
                    # "CPUExecutionProvider"
                ]
            )
            SessionManager.sessList[model_name] = session
            SessionManager.inputList[model_name] = {
                session.get_inputs()[0].name: np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            }

        return SessionManager.sessList[model_name]
    
    @staticmethod
    def run_model(model_name: str):
        if model_name not in SessionManager.sessList.keys():
            return f"Model {model_name} is not loaded"
        
        session = SessionManager.sessList[model_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # height, width = (1, 3, 640, 640)
        inputs = SessionManager.inputList[model_name]
        from time import time_ns
        start_time = time_ns()
        outputs = session.run([output_name], inputs)
        exec_time = time_ns() - start_time
        output = f"{model_name} output: {exec_time / (10 ** 6)} ms, {outputs[0].shape}"
        
        return output