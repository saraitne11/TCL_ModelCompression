import subprocess
import psutil
import argparse
import logging
import os
import signal
import time
import datetime

from logging.handlers import RotatingFileHandler
from typing import List, Dict


SIG_KILL = False


def signal_handler(signum, frame):
    global SIG_KILL
    SIG_KILL = True
    print('receive signal', signum)
    return


def get_proc_memory(pid: int) -> int:
    """
    :param pid: Process ID
    :return: MiB
    """
    p = psutil.Process(pid)
    mem = p.memory_info().rss / 2 ** 20
    return int(mem)


def get_gpu_proc_info() -> List[Dict]:
    """
    :return: [
        {'PNAME': pname(str), 'PID': pid(int), 'GPU_MEM': xxx(int, MiB), 'MEM': xxx(int, MiB)},
        {'PNAME': pname(str), 'PID': pid(int), 'GPU_MEM': xxx(int, MiB), 'MEM': xxx(int, MiB)},
        ...
    ]
    """
    cmd = ['nvidia-smi', '--query-compute-apps=process_name,pid,used_gpu_memory', '--format=csv,noheader']
    out = subprocess.check_output(cmd)
    out = out.decode().split('\n')

    res = []
    for _line in out:
        if _line:
            if 'MiB' not in _line:
                _line = _line.replace('[N/A]', '0 MiB')
            items = list(map(str.strip, _line.split(',')))
            pname = items[0]
            pid = int(items[1])
            _s = items[2]
            p_gpu_mem = int(_s[:_s.index(' MiB')])
            p_mem = get_proc_memory(pid)

            r = {
                'PNAME': pname,
                'PID': pid,
                'GPU_MEM': p_gpu_mem,
                'MEM': p_mem
            }
            res.append(r)
    return res


def sum_proc_info(proc_info: List[Dict], _key: str) -> int:
    return sum(map(lambda x: x[_key], proc_info))


def max_proc_info(proc_info: List[Dict], _key: str) -> Dict:
    return sorted(proc_info, key=lambda x: x[_key], reverse=True)[0]


def get_logging_str() -> str:
    proc_info = get_gpu_proc_info()
    _max = max_proc_info(proc_info, 'GPU_MEM')
    total_gpu_mem = sum_proc_info(proc_info, 'GPU_MEM')
    total_mem = sum_proc_info(proc_info, 'MEM')

    s = f"PNAME: {_max['PNAME']}, PID: {_max['PID']}, " \
        f"GPU_MEM: {_max['GPU_MEM']}({total_gpu_mem}), " \
        f"MEM: {_max['MEM']}({total_mem})"
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--log_period', type=int, default=1,
                        help="Logging Period (Second)")
    parser.add_argument('--log_file_size', type=int, default=10,
                        help="Target Log File Rotating Size (MiB)")
    parser.add_argument('--log_file_count', type=int, default=5,
                        help="Target Log File Backup Count")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    logger = logging.getLogger('FLASK')
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(args.log_file,
                                  mode='a',
                                  maxBytes=1024 * 1024 * args.log_file_size,
                                  backupCount=args.log_file_count)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging_time = time.time() + args.log_period
    while True:
        cur_time = time.time()
        if logging_time < cur_time:
            logger.info(get_logging_str())
            logging_time = cur_time + args.log_period

        time.sleep(1)
        if SIG_KILL:
            break
