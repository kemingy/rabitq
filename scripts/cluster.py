from struct import unpack, pack
from sys import argv
from functools import partial

from faiss import Kmeans
import numpy as np
from tqdm import tqdm


def reservoir_sampling(iterator, k: int):
    """Reservoir sampling from an iterator."""
    res = []
    while len(res) < k:
        res.append(next(iterator))
    for i, vec in enumerate(iterator, k + 1):
        j = np.random.randint(0, i)
        if j < k:
            res[j] = vec
    return res


def default_filter(i, vec):
    return True


def read_vec_yield(
    filepath: str, vec_type: np.dtype = np.float32, picker=default_filter
):
    """Read vectors and yield an iterator."""
    size = np.dtype(vec_type).itemsize
    i = 0
    with open(filepath, "rb") as f:
        while True:
            try:
                buf = f.read(4)
                if len(buf) == 0:
                    break
                dim = unpack("<i", buf)[0]
                vec = np.frombuffer(f.read(dim * size), dtype=vec_type)
                if picker(i, vec):
                    yield vec
            except Exception as err:
                print(err)
                break
            i += 1


def write_vec(filepath: str, vecs: np.ndarray, vec_type: np.dtype = np.float32):
    """Write vectors to a file. Support `fvecs`, `ivecs` and `bvecs` format."""
    with open(filepath, "wb") as f:
        for vec in vecs:
            f.write(pack("<i", len(vec)))
            f.write(vec.tobytes())


def inspect_vecs_file_dim(filename: str) -> int:
    with open(filename, "rb") as f:
        buf = f.read(4)
        dim = unpack("<i", buf)[0]
        return dim


def hierarchical_kmeans(filename: str, n_cluster_top: int, n_cluster_down: int):
    dim = inspect_vecs_file_dim(filename)
    vecs = np.fromiter(read_vec_yield(filename), dtype=np.dtype((float, dim)))
    top = Kmeans(dim, n_cluster_top)
    top.train(vecs)
    _, labels = top.assign(vecs)

    centroids = []
    for i in range(n_cluster_top):
        down = Kmeans(dim, n_cluster_down)
        down.train(vecs[labels == i])
        centroids.append(down.centroids)

    return np.vstack(centroids)


def hierarchical_kmeans_with_sampling(
    filename, n_cluster_top, n_cluster_down, max_point_per_cluster=256
):
    top_points = reservoir_sampling(
        read_vec_yield(filename), n_cluster_top * max_point_per_cluster
    )
    dim = top_points[0].shape[0]

    top_cluster = Kmeans(dim, n_cluster_top)
    top_cluster.train(top_points)

    labels = np.zeros(1_000_000, dtype=np.uint32)
    for i, vec in tqdm(enumerate(read_vec_yield(filename)), desc="assign labels"):
        _, label = top_cluster.assign(vec.reshape((1, -1)))
        labels[i] = label[0]

    def filter_label(label, i, vec):
        return labels[i] == label

    centroids = []
    for i in tqdm(range(n_cluster_top), desc="down clustering"):
        down_points = reservoir_sampling(
            read_vec_yield(filename, picker=partial(filter_label, i)),
            n_cluster_down * max_point_per_cluster,
        )
        down_cluster = Kmeans(dim, n_cluster_down)
        down_cluster.train(down_points)
        centroids.append(down_cluster.centroids)

    write_vec(f"centroids_{n_cluster_top}_{n_cluster_down}.fvecs", np.vstack(centroids))


if __name__ == "__main__":
    filename = argv[1]
    top_n = int(argv[2])
    down_n = int(argv[3])
    hierarchical_kmeans_with_sampling(filename, top_n, down_n)
