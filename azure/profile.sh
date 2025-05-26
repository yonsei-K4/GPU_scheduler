#!/bin/bash

# Build 단계는 주석 처리 (필요 시 해제)
# ./build.sh --config Release --update --build --skip_tests \
#     --use_cuda --enable_cuda_profiling \
#     --cmake_extra_defines onnxruntime_ENABLE_NVTX_PROFILE=ON \
#     --cmake_extra_defines onnxruntime_DISABLE_CONTRIB_OPS=ON \
#     --cmake_extra_defines onnxruntime_BUILD_SHARED_LIB=ON \
#     --allow_running_as_root \
#     --parallel $(nproc) --compile_no_warning_as_error --build_wheel

# ----------------------------
# NSYS 프로파일링
# ----------------------------
echo "[+] Running nsys profile..."
nsys profile -o run \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    python bench.py

# ----------------------------
# NSYS -> SQLite 변환
# ----------------------------
echo "[.] Converting nsys report to sqlite..."
nsys export --type sqlite --force-overwrite=true -o run.sqlite run.nsys-rep

# ----------------------------
# ONNX 노드 - CUDA 커널 매핑
# ----------------------------
# echo "[.] Mapping ONNX Runtime symbols to kernel symbols..."
# sqlite3 -header -csv run.sqlite "
# SELECT
#     nvtx.text                              AS onnx_node,
#     ker.gridId                             AS kernel_id,
#     sid.value                              AS cuda_kernel,
#     ROUND((ker.end - ker.start)/1e6, 3)    AS time_ms
# FROM   NVTX_EVENTS               nvtx
# JOIN   CUPTI_ACTIVITY_KIND_KERNEL ker
#            ON ker.start BETWEEN nvtx.start AND nvtx.end
# JOIN   StringIds                 sid
#            ON sid.id = ker.shortName
# WHERE  nvtx.eventType = 60
#   AND  nvtx.text != 'Batch- Forward'
# ORDER  BY nvtx.start, ker.start;
# " > nsys_trace.csv

# ----------------------------
# NCU (커널 단위 프로파일링)
# ----------------------------
# echo "[+] Running ncu profile..."
# ncu --target-processes all \
#     --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
#     --csv \
#     --log-file ncu_prof.csv \
#     python bench.py

    # --section LaunchStats \
