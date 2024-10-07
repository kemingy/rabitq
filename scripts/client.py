from time import perf_counter
from struct import unpack, pack
import concurrent.futures
from argparse import ArgumentParser

import httpx
import numpy as np
from tqdm import tqdm


def read_vec(filepath: str, vec_type: np.dtype = np.float32):
    """Read vectors from a file. Support `fvecs`, `ivecs` and `bvecs` format.
    Args:
        filepath: The path of the file.
        vec_type: The type of the vectors.
    """
    size = np.dtype(vec_type).itemsize
    with open(filepath, "rb") as f:
        vecs = []
        while True:
            try:
                buf = f.read(4)
                if len(buf) == 0:
                    break
                dim = unpack("<i", buf)[0]
                vecs.append(np.frombuffer(f.read(dim * size), dtype=vec_type))
            except Exception as err:
                print(err)
                break
    return np.array(vecs)


def write_vec(filepath: str, vecs: np.ndarray, vec_type: np.dtype = np.float32):
    """Write vectors to a file. Support `fvecs`, `ivecs` and `bvecs` format."""
    with open(filepath, "wb") as f:
        for vec in vecs:
            f.write(pack(len(vec), "<i"))
            f.write(vec.tobytes())


def recall(truth, res, topk):
    count = 0
    length = min(topk, len(truth))
    for t in truth[:length]:
        if t in res[:length]:
            count += 1
    return count / float(length)


def single_client_query(queries, truths, limit=1000, top_k=100, nprobe=300):
    r = 0
    num = 0
    total_latency = 0
    with httpx.Client(timeout=3000) as client:
        for query, truth in tqdm(zip(queries, truths)):
            resp, latency = query_vec(client, query, truth, top_k=top_k, probe=nprobe)
            if resp is None:
                continue
            r += resp
            total_latency += latency
            num += 1
            if num >= limit:
                break

    print(
        f"recall: {r / num:.6}, avg latency: {total_latency / num:.6}"
        f" [top_k={top_k}, nprobe={nprobe}]"
    )


def query_vec(client, query, truth, top_k=100, probe=300):
    start = perf_counter()
    resp = client.post(
        "http://127.0.0.1:9000/query",
        json={
            "query": query.tolist(),
            "top_k": top_k,
            "probe": probe,
        },
    )
    if resp.status_code != 200:
        print(resp.content)
        return None
    return recall(truth, resp.json()["ids"], top_k), perf_counter() - start


def concurrent_client_query(queries, truths, limit=1000, top_k=100, nprobe=300):
    r = 0
    num = 0
    total_latency = 0
    client = httpx.Client(timeout=3000)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(query_vec, client, query, truth, top_k, nprobe)
            for (query, truth) in list(zip(queries, truths))[:limit]
        ]
        for future in tqdm(concurrent.futures.as_completed(futures)):
            resp = future.result()
            if resp is None:
                continue
            r += resp[0]
            total_latency += resp[1]
            num += 1
    client.close()
    print(
        f"recall: {r / num:.6}, avg latency: {total_latency / num:.6}"
        f"[top_k={top_k}, nprobe={nprobe}]"
    )


def build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--limit", "-l", type=int, default=1000)
    parser.add_argument("--concurrent", "-c", action="store_true")
    parser.add_argument("--top_k", "-k", type=int, default=100)
    parser.add_argument("--nprobe", "-p", type=int, default=300)
    parser.add_argument("--query_file", "-q", default="./gist/gist_query.fvecs")
    parser.add_argument("--truth_file", "-t", default="./gist/gist_groundtruth.ivecs")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    queries = read_vec(args.query_file, vec_type=np.float32)
    truths = read_vec(args.truth_file, vec_type=np.int32)

    if args.concurrent:
        concurrent_client_query(
            queries, truths, limit=args.limit, top_k=args.top_k, nprobe=args.nprobe
        )
    else:
        single_client_query(
            queries, truths, limit=args.limit, top_k=args.top_k, nprobe=args.nprobe
        )
