import time

from utils import predict, prepare_input
from multiprocessing import Pool
from argparse import ArgumentParser


parser = ArgumentParser(description='Params')
parser.add_argument('--n_chunks', default=20, type=int, help='num parts to divide the dataset into')
parser.add_argument('--n_jobs', default=8, type=int, help='num processes to use')
parser.add_argument('--multiplier', default=2, type=int, help='how much data use')

args = parser.parse_args()


def get_batch(data):
    n = len(data)
    num_iters = n // args.n_chunks
    num_last = n % args.n_chunks
    if num_last > 0:
        num_iters += 1
    i = 0
    chunks = []
    for c in range(num_iters):
        if c == num_iters - 1 and num_last > 0:
            curr_data = data[i: i + num_last]
        else:
            curr_data = data[i:i + args.n_chunks]
            i += args.n_chunks

        chunks.append(curr_data)

    return chunks


def run_multi():
    data = prepare_input(multiplier=args.multiplier)
    chunks = get_batch(data)
    pool = Pool(args.n_jobs)
    time_start = time.time()
    results = []
    for res in pool.imap_unordered(predict, chunks):
        results.append(res)
    pool.close()
    pool.join()

    time_finish = time.time()

    print(f'Result with len data: {len(data)}, n_jobs={args.n_jobs}, n_chunks={args.n_chunks}, time = {time_finish - time_start}')
    return results


def run_single():
    data = prepare_input(multiplier=args.multiplier)
    time_start = time.time()
    _ = predict(data)
    time_result = time.time() - time_start

    print(f'Result with len data: {len(data)}, n_jobs=1, n_chunks=1, time = {time_result}')


if __name__ == '__main__':
    run_single()
    run_multi()
