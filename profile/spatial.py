# run_parallel.py
import os, json, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from session import SessionManager   # 질문에 주신 session.py
import onnx
from onnx import NodeProto

ITER = 1000                     # ⬅️ 모델당 반복 횟수
MAX_WORKERS = 32               # GPU 당 점유 SM 수에 맞춰 조정

def contains_loop(model_path: str) -> bool:
    g = onnx.load(model_path).graph
    return any(n.op_type == "Loop" for n in g.node)
###############################################################################
# 1. 모델 목록 로드
###############################################################################
ROOT = os.path.dirname(__file__)
with open(os.path.join(ROOT, "models.json")) as f:
    model_list = json.load(f).get("models", [])

if not model_list:
    raise RuntimeError("models.json 이 비어 있습니다.")

###############################################################################
# 2. 세션 로드 (세션 ≒ 독립 CUDA 스트림)
###############################################################################
providers = dict()  # 모델별 프로바이더 설정}

for m in model_list:
    has_loop = contains_loop(os.path.join('/', "models", f"{m}.onnx"))

    if has_loop: 
        prov = ("CUDAExecutionProvider", {"device_id": 0})              # 기본 설정
    else:
        prov = ("CUDAExecutionProvider", {"device_id": 0,
                                        "do_copy_in_default_stream": 0})
    providers[m] = prov

print("🚀 모델 로딩 중...")
for m in model_list:
    if SessionManager.fetch_model(m, providers[m], batch_size=1) is None:
        raise RuntimeError(f"{m} 로드 실패")
    else:
        print(f"✅ {m} 로드 완료")

# 워밍업
for m in model_list:
    SessionManager.run_model(m)

###############################################################################
# 3. 병렬 추론 워커
###############################################################################
def worker(model_name: str, iters: int):
    lat = []
    for k in range(iters):
        torch.cuda.nvtx.range_push(f"{model_name}_iter{k}")
        out = SessionManager.run_model(model_name)
        torch.cuda.nvtx.range_pop()

        ms = float(out.split("output:")[1].split("ms")[0].strip())
        lat.append(ms)
    return model_name, lat


torch.cuda.cudart().cudaProfilerStart()
t0 = time.time()


lat_dict = defaultdict(list)
with ThreadPoolExecutor(max_workers=len(model_list)) as pool:
    futures = [pool.submit(worker, m, ITER) for m in model_list]
    for f in tqdm(as_completed(futures), total=len(futures), unit="model"):
        name, l = f.result()
        lat_dict[name].extend(l)

torch.cuda.cudart().cudaProfilerStop()
elapsed = time.time() - t0
total_jobs = len(model_list) * ITER


print(f"\n✅ {total_jobs} 회 추론 완료 – {elapsed:.2f}s, "
      f"{total_jobs/elapsed:.1f} infer/s\n")
for m in model_list:
    l = lat_dict[m]
    print(f"🔹 {m:20s}  평균 {np.mean(l):7.2f} ms  |  σ {np.std(l):6.2f} ms")