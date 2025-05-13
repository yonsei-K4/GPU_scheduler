import pynvml
import time

def initialize_pynvml():
    """Initialize the pynvml library."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Failed to initialize pynvml: {e}")
        exit(1)

def get_gpu_count():
    """Return the number of GPUs available."""
    return pynvml.nvmlDeviceGetCount()

def get_gpu_info(handle):
    """Get utilization and memory info for a GPU handle."""
    # Get utilization rates (GPU compute usage in %)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu  # Compute utilization

    # Get memory info
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used / 1024**2  # Convert to MiB
    mem_total = mem_info.total / 1024**2  # Convert to MiB
    mem_percent = (mem_used / mem_total) * 100

    return gpu_util, mem_used, mem_total, mem_percent

def print_gpu_stats():
    """Print utilization and memory stats for all GPUs."""
    initialize_pynvml()
    gpu_count = get_gpu_count()
    
    print(f"Found {gpu_count} GPU(s)")
    print("GPU Index | GPU Util (%) | Mem Used (MiB) | Mem Total (MiB) | Mem Usage (%)")
    print("-" * 70)

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        gpu_util, mem_used, mem_total, mem_percent = get_gpu_info(handle)
        
        print(f"{i:9} | {gpu_util:11} | {mem_used:13.2f} | {mem_total:13.2f} | {mem_percent:12.2f}")

    pynvml.nvmlShutdown()

def monitor_gpus(interval=1, duration=10):
    """Monitor GPUs at specified intervals for a given duration."""
    initialize_pynvml()
    gpu_count = get_gpu_count()
    
    for _ in range(int(duration / interval)):
        print(f"\nTimestamp: {time.ctime()}")
        print("GPU Index | GPU Util (%) | Mem Used (MiB) | Mem Total (MiB) | Mem Usage (%)")
        print("-" * 70)
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            gpu_util, mem_used, mem_total, mem_percent = get_gpu_info(handle)
            
            print(f"{i:9} | {gpu_util:11} | {mem_used:13.2f} | {mem_total:13.2f} | {mem_percent:12.2f}")
        
        time.sleep(interval)
    
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    # Option 1: Print stats once
    print_gpu_stats()
    
    # Option 2: Monitor continuously for 10 seconds with 1-second intervals
    monitor_gpus(interval=1, duration=10)