import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("gpu_metrics.csv")

zero_timestamps = df[
    (df["metricName"] == "Compute Warps in Flight [Throughput %]") & (df["value"] == 0)
]["timestamp"].unique()

# 해당 timestamp를 포함하는 모든 행 제거
df = df[~df["timestamp"].isin(zero_timestamps)]


read_df = df[df["metricName"] == "DRAM Read Bandwidth [Throughput %]"]
write_df = df[df["metricName"] == "DRAM Write Bandwidth [Throughput %]"]

# timestamp 기준으로 merge 후 value 합산
dram_df = pd.merge(read_df, write_df, on="timestamp", suffixes=("_read", "_write"))
dram_df["value"] = dram_df["value_read"] + dram_df["value_write"]
dram_df["metricName"] = "DRAM Bandwidth [Throughput %]"

# 필요한 컬럼만 정리
dram_df = dram_df[["rawTimestamp_read", "timestamp", "timestamp_sec_read", "metricId_read", "metricName", "value"]]
dram_df.columns = ["rawTimestamp", "timestamp", "timestamp_sec", "metricId", "metricName", "value"]

# 기존 Compute metric만 남기고 결합
compute_df = df[df["metricName"] == "Compute Warps in Flight [Throughput %]"]
combined_df = pd.concat([compute_df, dram_df], ignore_index=True)

# metricName별 통계 요약
summary = combined_df.groupby("metricName")["value"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
).reset_index()

# 결과 출력
print(summary)

combined_df.to_csv("result.csv", index=False)

# # 그래프 그리기 (선택 사항)
# import matplotlib.pyplot as plt

# for metric in df["metricName"].unique():
#     subset = df[df["metricName"] == metric]


#     plt.plot(subset["timestamp_sec"], subset["value"], label=metric)

# plt.xlabel("Time (s)")
# plt.ylabel("Metric Value")
# plt.legend()
# plt.title("GPU Metric Trends Over Time")
# plt.grid(True)
# plt.show()
