from jtop import jtop
import argparse
import logging
import os
import signal
import time

from logging.handlers import RotatingFileHandler


SIG_KILL = False


def signal_handler(signum, frame):
    global SIG_KILL
    SIG_KILL = True
    print('receive signal', signum)
    return


def main():
    parser = argparse.ArgumentParser()
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

    logger = logging.getLogger('JTOP_MONITOR')
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
    with jtop() as jetson:
        while True:
            cur_time = time.time()
            if logging_time < cur_time:
                d = jetson.ram
                cpu_mem = (d['use'] - d['shared']) * 10 ** 3    # CPU Memory (Byte)
                cpu_mem = int(cpu_mem / 2 ** 20)                # CPU Memory (MiB)

                gpu_mem = d['shared'] * 10 ** 3                 # GPU Memory (Byte)
                gpu_mem = int(gpu_mem / 2 ** 20)                # GPU Memory (MiB)

                logger.info(f"Memory: {cpu_mem} MiB, GPU Memory: {gpu_mem} MiB")
                logging_time = cur_time + args.log_period

            time.sleep(1)
            if SIG_KILL:
                break


if __name__ == '__main__':
    main()
