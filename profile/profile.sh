#!/bin/bash

# build ONNX Runtime with CUDA profiling enabled
# ./build.sh   --config Release --update --build --skip_tests --use_cuda --enable_cuda_profiling --cmake_extra_defines  onnxruntime_ENABLE_NVTX_PROFILE=ON onnxruntime_DISABLE_CONTRIB_OPS=ON onnxruntime_BUILD_SHARED_LIB=ON --allow_running_as_root   --parallel $(nproc) --compile_no_warning_as_error --build_wheel

# run nsys profile
echo "[+] Running nsys profile..."
nsys profile -o run --trace=cuda,nvtx,osrt --sample=process-tree --gpu-metrics-devices=all --gpu-metrics-frequency=1000 --force-overwrite=true python run.py

# convert nsys report to sqlite
echo "[.] Converting nsys report to sqlite..."
nsys export --type sqlite --force-overwrite=true -o run.sqlite run.nsys-rep

# map onnxruntime symbols to kernel symbols
echo "[.] Mapping ONNX Runtime symbols to kernel symbols..."
sqlite3 -header -csv run.sqlite "
SELECT
    nvtx.text                              AS onnx_node,
    ker.gridId                             AS kernel_id,
    sid.value                              AS cuda_kernel,
    ROUND((ker.end - ker.start)/1e6, 3)    AS time_ms
FROM   NVTX_EVENTS               nvtx
JOIN   CUPTI_ACTIVITY_KIND_KERNEL ker
           ON ker.start BETWEEN nvtx.start AND nvtx.end
JOIN   StringIds                 sid
           ON sid.id = ker.shortName
WHERE  nvtx.eventType = 60
  AND  nvtx.text != 'Batch- Forward'
ORDER  BY nvtx.start, ker.start;
" > nsys_trace.csv

# profiling at kernel level
echo "[+] Running ncu profile..."
ncu --target-processes all \
    --section LaunchStats \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    --log-file ncu_prof.csv \
    python run.py

