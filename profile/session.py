# Global Session Manager
import os
import torch
import numpy as np
import onnxruntime

class SessionManager:
    sessList = {}
    inputList = {}
    batchList = {}
    def __init__():
        pass

    @staticmethod
    def fetch_model(model_name: str, providers: tuple, batch_size: int = 1):
        if model_name in SessionManager.sessList:
            return SessionManager.sessList[model_name]

        model_path = os.path.join(os.path.dirname(__file__), f"/models/{model_name}.onnx")
        SessionManager.batchList[model_name] = batch_size

        
            
        # 세션 생성
        session = onnxruntime.InferenceSession(
            model_path,
            providers=[providers]
        )
        SessionManager.sessList[model_name] = session

        if "mask" in model_name.lower():
            batch_size = 1  

        input_dict = {}

        for input_meta in session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype = input_meta.type

            # 동적 차원 해결
            resolved_shape = []
            # print(shape)
            for dim in shape:
                if isinstance(dim, str) and dim.startswith("batch"):
                    resolved_shape.append(batch_size)
                elif isinstance(dim, str) and dim in ["height", "width"]:
                    resolved_shape.append(640)  # 예시로 640으로 설정\
                elif isinstance(dim, str) and dim == "sequence":
                    resolved_shape.append(512)
                else:
                    resolved_shape.append(dim)

            # dtype 처리
            print(resolved_shape, dtype)
            if "float" in dtype:
                arr = np.random.randn(*resolved_shape).astype(np.float32)
            elif "int64" in dtype:
                arr = np.random.randint(0, 100, size=resolved_shape, dtype=np.int64)
            elif "int32" in dtype:
                arr = np.random.randint(0, 100, size=resolved_shape, dtype=np.int32)
            else:
                raise ValueError(f"❌ '{name}' 입력의 지원되지 않는 타입: {dtype}")

            input_dict[name] = arr

        SessionManager.inputList[model_name] = input_dict
        return session

    @staticmethod
    def run_model(model_name: str):
        session = SessionManager.sessList[model_name]
        output_name = session.get_outputs()[0].name
        

        # height, width = (1, 3, 640, 640)
        inputs = SessionManager.inputList[model_name]
        from time import time_ns

        torch.cuda.nvtx.range_push(f"{model_name}_inference")
        start_time = time_ns()
        outputs = session.run([output_name], inputs)
        exec_time = time_ns() - start_time
        torch.cuda.nvtx.range_pop()
        
        output = f"{model_name} output: {exec_time / (10 ** 6)} ms, {outputs[0].shape}"
        
        return output