import os
import json
import torch
from tqdm import tqdm
import numpy as np
# from profile.runner import ModelRunner
from session import SessionManager
from collections import defaultdict

iter = 1000

if __name__ == "__main__":
    import json
    import os
    import time

    print("🚀 [Standalone] 모델 로딩 및 추론 시작")

    # 모델 목록 로드
    model_file_path = os.path.join(os.path.dirname(__file__), "models.json")
    try:
        with open(model_file_path, "r") as f:
            model_data = json.load(f)
            model_list = model_data.get("models", [])
    except Exception as e:
        print(f"❌ 모델 목록 로딩 실패: {e}")
        exit(1)

    if not model_list:
        print("❌ 모델 목록이 비어 있습니다.")
        exit(1)

    # Provider 설정
    providers = {m: ("CUDAExecutionProvider", {"device_id": 0}) for m in model_list}

    # 모델 로딩
    for model_name in model_list:
        session = SessionManager.fetch_model(model_name, providers[model_name], batch_size=1)
        if session is None:
            print(f"❌ 모델 {model_name} 로드 실패")
            continue
        else:
            print(f"✅ 모델 {model_name} 로드 완료")

    # 추론 실행
    latency_stats = defaultdict(list)  # 모델별 latency(ms) 저장


    # 미리 한번 실행하여 메모리에 올려놓기
    for model_name in model_list:
            result = SessionManager.run_model(model_name)

    print("\n🧠 [INFERENCE] 모델별 추론 시작")
    start_time = time.time()
    torch.cuda.cudart().cudaProfilerStart()

    for model_name in tqdm(model_list * iter, desc="모델 추론 진행 중", unit="model"):
        # print(f"\n🔍 {model_name} 추론 중...")
        result = SessionManager.run_model(model_name)
        if isinstance(result, str) and "output:" in result:
            try:
                latency_str = result.split("output:")[1].split("ms")[0].strip()
                latency = float(latency_str)
                latency_stats[model_name].append(latency)
            except Exception as e:
                print(f"[WARN] latency 파싱 실패: {result}")
        else:
            print(f"[WARN] 예상과 다른 결과 형식: {result}")

        # time.sleep(0.2)  # profiling 시 편의 위해 간격 삽입

    torch.cuda.cudart().cudaProfilerStop()
    end_time = time.time()
    total_time = end_time - start_time

    # 추론 횟수 및 throughput 계산
    total_inferences = len(model_list) * iter
    throughput = total_inferences / total_time

    print("\n✅ 모든 추론 완료")
    print(f"⏱️ 전체 소요 시간: {total_time:.3f}초")
    print(f"📈 Inference Throughput: {throughput:.2f} inferences/sec")

    # 모델별 latency 통계 출력
    print("\n📊 모델별 Latency 통계:")
    for model_name in model_list:
        latencies = latency_stats.get(model_name, [])
        if latencies:
            mean = np.mean(latencies)
            std = np.std(latencies)
            print(f"🔹 {model_name}: 평균 {mean:.3f} ms, 표준편차 {std:.3f} ms, 횟수 {len(latencies)}회")
        else:
            print(f"⚠️ {model_name}: latency 데이터 없음")