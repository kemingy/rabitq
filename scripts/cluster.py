from struct import unpack, pack
from sys import argv
from functools import partial

from faiss import Kmeans
import numpy as np
from tqdm import tqdm


def default_filter(vec):
    return True


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


def read_vec_yield(
    filepath: str, vec_type: np.dtype = np.float32, filter=default_filter
):
    """Read vectors and yield an iterator."""
    size = np.dtype(vec_type).itemsize
    with open(filepath, "rb") as f:
        while True:
            try:
                buf = f.read(4)
                if len(buf) == 0:
                    break
                dim = unpack("<i", buf)[0]
                vec = np.frombuffer(f.read(dim * size), dtype=vec_type)
                if filter(vec):
                    yield vec
            except Exception as err:
                print(err)
                break


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
            f.write(pack("<i", len(vec)))
            f.write(vec.tobytes())


def hierarchical_kmeans(vecs, n_cluster_top, n_cluster_down):
    dim = vecs.shape[1]
    top = Kmeans(dim, n_cluster_top)
    top.train(vecs)
    _, labels = top.assign(vecs)

    centroids = []
    for i in range(n_cluster_top):
        down = Kmeans(dim, n_cluster_down)
        down.train(vecs[labels == i])
        centroids.append(down.centroids)

    return np.vstack(centroids)


if __name__ == "__main__":
    filename = argv[1]
    top_n = int(argv[2])
    down_n = int(argv[3])
    max_point_per_cluster = 256
    top_points = reservoir_sampling(
        read_vec_yield(filename), top_n * max_point_per_cluster
    )
    dim = top_points[0].shape[0]

    top_cluster = Kmeans(dim, top_n)
    top_cluster.train(top_points)

    def filter_label(label, vec):
        _, label = top_cluster.assign(vec.reshape((1, -1)))
        return label[0] == label

    centroids = []
    for i in tqdm(range(top_n)):
        down_points = reservoir_sampling(
            read_vec_yield(filename, filter=partial(filter_label, i)),
            down_n * max_point_per_cluster,
        )
        down_cluster = Kmeans(dim, down_n)
        down_cluster.train(down_points)
        centroids.append(down_cluster.centroids)

    write_vec(f"centroids_{top_n}_{down_n}.fvecs", np.vstack(centroids))
