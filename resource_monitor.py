import subprocess
import psutil

from typing import Tuple


def get_proc_memory(pid: int) -> int:
    p = psutil.Process(pid)
    return p.memory_info().rss / 2 ** 20


def get_gpu_proc_memory() -> Tuple[str, int, int, int]:
    """
    :return: (Process Name, PID, Memory Usage, GPU Memory Usage)
    """
    cmd = ['nvidia-smi', '--query-compute-apps=process_name,pid,used_gpu_memory', '--format=csv,noheader']
    out = subprocess.check_output(cmd)
    total_usage = 0
    for _line in out.decode().split('\n'):
        if 'MiB' in _line:

            proc_usage = _line[0:_line.index(' MiB')]
            total_usage += int(proc_usage)
    return


gpu_mem_usage = get_gpu_proc_memory()
print(gpu_mem_usage)
