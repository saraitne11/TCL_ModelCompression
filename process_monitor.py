import subprocess
import psutil
import argparse
import logging
import os
import signal
import time

from logging.handlers import RotatingFileHandler
from typing import Tuple


SIG_KILL = False


def signal_handler(signum, frame):
    global SIG_KILL
    SIG_KILL = True
    print('receive signal', signum)
    return


def get_proc_memory(_p: psutil.Process) -> int:
    return int(_p.memory_info().rss / 2 ** 20)


def get_gpu_proc_info(_pid: int) -> Tuple[str, int]:
    """
    :return: {'PNAME': pname(str), 'PID': pid(int), 'GPU_MEM': xxx(int, MiB), 'MEM': xxx(int, MiB)}
    """
    cmd = ['nvidia-smi', '--query-compute-apps=process_name,pid,used_gpu_memory', '--format=csv,noheader']
    out = subprocess.check_output(cmd)
    out = out.decode().split('\n')
    proc_info = list(filter(lambda x: str(_pid) in x, out))
    if not proc_info:
        return '', 0

    proc_info = proc_info[0]
    if 'MiB' not in proc_info:
        proc_info = proc_info.replace('[N/A]', '0 MiB')
    items = list(map(str.strip, proc_info.split(',')))
    pname = items[0]
    _s = items[2]
    gpu_mem = int(_s[:_s.index(' MiB')])
    return pname, gpu_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-pid', required=True, type=int,
                        help="Logging Target Process ID")
    parser.add_argument('--log-file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--log-period', type=int, default=1,
                        help="Logging Period (Second), Default=1")
    parser.add_argument('--log-file-size', type=int, default=10,
                        help="Target Log File Rotating Size (MiB), Default=10")
    parser.add_argument('--log-file-count', type=int, default=5,
                        help="Target Log File Backup Count, Default=5")
    args = parser.parse_args()

    if os.path.dirname(args.log_file):
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    logger = logging.getLogger('PROCESS_MONITOR')
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

    pid = args.target_pid
    p = psutil.Process(pid)

    logging_time = time.time() + args.log_period
    while True:
        cur_time = time.time()
        if logging_time < cur_time:
            pname, gpu_mem = get_gpu_proc_info(pid)
            mem = get_proc_memory(p)
            logger.info(f"{pname}({pid}), Memory: {mem} MiB, GPU Memory: {gpu_mem} MiB")
            logging_time = cur_time + args.log_period

        time.sleep(1)
        if SIG_KILL:
            break


if __name__ == '__main__':
    main()
