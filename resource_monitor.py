import subprocess
import psutil


def get_process_used_gpu_memory(pid: int):
    cmd = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
    out = subprocess.check_output(cmd)
    out = list(filter(lambda x: str(pid) in x, out.decode().split('\n')))
    if len(out) == 1:
        return out[0]
    else:
        return ''


def get_process_cpu_memory(pid: int):
    p = psutil.Process(pid)
    print(p.memory_info().rss / 2 ** 20)
    print(p.cpu_percent())
    return


r = get_process_used_gpu_memory(12184)
print(r)
get_process_cpu_memory(12184)
