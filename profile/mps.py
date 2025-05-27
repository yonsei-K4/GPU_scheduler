# run_parallel_mps.py
import os, json, time, multiprocessing as mp
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch

from session import SessionManager
import onnx

ROOT = os.path.dirname(__file__)
ITER = 1000                    # 모델당 추론 횟수

###############################################################################
# 1) 모델 목록
###############################################################################
with open(os.path.join(ROOT, "models.json")) as f:
    model_list = json.load(f)["models"]
if not model_list:
    raise RuntimeError("models.json 이 비어 있습니다")

###############################################################################
# 2) 모델별 SM 지분(%)
###############################################################################
# model_shares = { m : int(100/len(model_list)) for m in model_list }  # 균등 분배
# print("모델별 MPS 지분 설정:", model_shares)
# model_shares = {               # version 1.0 not bad
#     "mask_rcnn" : 50,
#     "bert" : 5,
#     "resnet50" : 20,
#     "mobilenet_v2" : 10,
#     "inception_v4" : 10,
#     "squeezenet1_0" : 5, 
# }
model_shares = {               # 원하는 비율로 수정
    "mask_rcnn" : 45,
    "bert" : 3,
    "resnet50" : 12,
    "mobilenet_v2" : 10,
    "inception_v4" : 20,
    "squeezenet1_0" : 10, 
}
# DEFAULT_SHARE = 20

###############################################################################
# 3) Loop 포함 여부 → Copy-Stream 옵션 결정
###############################################################################
def contains_loop(path: str) -> bool:
    return any(n.op_type == "Loop" for n in onnx.load(path).graph.node)

def provider_for(model):
    mpth = os.path.join("/", "models", f"{model}.onnx")
    if contains_loop(mpth):
        return ("CUDAExecutionProvider", {"device_id": 0})
    return ("CUDAExecutionProvider", {"device_id": 0,
                                      "do_copy_in_default_stream": 0})

providers = {m: provider_for(m) for m in model_list}

###############################################################################
# 4) 워커 프로세스
###############################################################################
def worker(model, iters, share_pct, q):
    # ① 각 프로세스마다 MPS 지분 설정
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(share_pct)

    # ② 세션 로드 & 워밍업
    SessionManager.fetch_model(model, providers[model], batch_size=1)
    SessionManager.run_model(model)          # warm-up

    lat = []
    for k in range(iters):
        torch.cuda.nvtx.range_push(f"{model}_iter{k}")
        out = SessionManager.run_model(model)
        torch.cuda.nvtx.range_pop()

        ms = float(out.split("output:")[1].split("ms")[0].strip())
        lat.append(ms)

    q.put((model, lat))       # 결과 반환

###############################################################################
# 5) 메인 – 프로세스 병렬 실행
###############################################################################
if __name__ == "__main__":
    mp.set_start_method("spawn")             # CUDA 안전모드
    q = mp.Queue()
    procs = []
    t0 = time.time()                         # --------- ⬅️ 전체 타이머 시작
    torch.cuda.cudart().cudaProfilerStart()

    for m in model_list:
        pct = model_shares[m]
        p = mp.Process(target=worker, args=(m, ITER, pct, q))
        p.start()
        procs.append(p)

    # 진행바
    for p in tqdm(procs, desc="모델 프로세스 가동"):
        p.join()

    torch.cuda.cudart().cudaProfilerStop()

    wall_elapsed = time.time() - t0          # --------- ⬅️ 전체 타이머 종료

    # 결과 집계
    lat_dict = defaultdict(list)
    while not q.empty():
        name, lst = q.get()
        lat_dict[name].extend(lst)

    total_jobs = len(model_list) * ITER
    throughput = total_jobs / wall_elapsed

    print(f"\n✅ {total_jobs} 회 추론 완료 – {wall_elapsed:.2f}s, "
          f"{throughput:.1f} infer/s\n")

    for m in model_list:
        l = lat_dict[m]
        if l:
            print(f"🔹 {m:20s}  평균 {np.mean(l):7.2f} ms | σ {np.std(l):6.2f} ms | "
                  f"SM {model_shares[m]} %")
        else:
            print(f"⚠️ {m}: latency 수집 실패")