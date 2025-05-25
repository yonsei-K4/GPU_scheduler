import pandas as pd

# 파일 로드
ncu_df = pd.read_csv("ncu_prof.csv", skiprows=2)
nsys_df = pd.read_csv("nsys_trace.csv", header=None, names=["onnx_node", "kernel_id", "cuda_kernel", "time_ms"])

# 결과 저장용 리스트
merged = []

# 순차적으로 nsys의 커널 이름을 NCU의 Kernel Name에 매핑
ncu_kernels = ncu_df["Kernel Name"].dropna().unique().tolist()
print(f"NCU 커널 개수: {len(ncu_kernels)}")
ncu_idx = 0

for _, nsys_row in nsys_df.iterrows():
    cuda_kernel = nsys_row["cuda_kernel"]
    matched = False

    # ncu 커널들 중 순차적으로 하나씩 검사
    while ncu_idx < len(ncu_kernels):
        kernel_name = ncu_kernels[ncu_idx]
        ncu_idx += 1

        # 매칭: 부분 문자열 포함으로 처리
        if cuda_kernel in kernel_name or kernel_name in cuda_kernel:
            # NCU의 같은 커널 이름에 해당하는 모든 Metric들 가져오기
            ncu_rows = ncu_df[ncu_df["Kernel Name"] == kernel_name]

            for _, ncu_row in ncu_rows.iterrows():
                merged.append({
                    "onnx_node": nsys_row["onnx_node"],
                    "kernel_id": nsys_row["kernel_id"],
                    "cuda_kernel": cuda_kernel,
                    "ncu_kernel_name": kernel_name,
                    "ncu_metric_name": ncu_row["Metric Name"],
                    "ncu_metric_value": ncu_row["Metric Value"],
                    "ncu_metric_unit": ncu_row["Metric Unit"],
                    "ncu_section": ncu_row["Section Name"],
                    "time_ms": nsys_row["time_ms"]
                })
            matched = True
            break

    if not matched:
        merged.append({
            "onnx_node": nsys_row["onnx_node"],
            "kernel_id": nsys_row["kernel_id"],
            "cuda_kernel": cuda_kernel,
            "ncu_kernel_name": None,
            "ncu_metric_name": None,
            "ncu_metric_value": None,
            "ncu_metric_unit": None,
            "ncu_section": None,
            "time_ms": nsys_row["time_ms"]
        })

# CSV로 저장
merged_df = pd.DataFrame(merged)
merged_df.to_csv("merged_output.csv", index=False)
