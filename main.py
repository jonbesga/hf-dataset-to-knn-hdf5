from datasets import load_dataset
import h5py
import typer
import logging
import faiss
import torch

import warnings
warnings.simplefilter("ignore", UserWarning)

import faiss.contrib.torch_utils

logging.basicConfig(level=logging.INFO, format='%(message)s')

logger = logging.getLogger(__name__)


def bytes_to_gb(bytes_value):
    gb_value = bytes_value / (1024 ** 3)
    return gb_value

def format_millions(number):
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M".replace(".0M", "M")
    return str(number)

def format_dataset(name):
    return name.replace("/", "_")


def main(dataset_name:str, train_size: int = 1_000_000, test_size: int = 10_000, k: int = 100):
    assert k < train_size, "k must be less than or equal to the number of training points"

    ds = load_dataset(dataset_name, split=f"train[:{train_size}]", columns=["emb"], num_proc=16)
    ds = ds.with_format("np")
    
    X = torch.from_numpy(ds["emb"])

    logger.info("Creating GPU index")
    gpu_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), X.shape[1]) # euclidean

    logger.info("Adding vectors to GPU Index")
    X_gpu = X.cuda()
    gpu_index.add(X_gpu)

    Y = X[:test_size]
    Y_gpu = Y.cuda()
    
    logger.info("Vectors added. Searching...")
    distances = torch.zeros(test_size, k, dtype=torch.float32)
    indices = torch.zeros(test_size, k, dtype=torch.int64)
    gpu_index.search(Y_gpu, k, distances, indices)
    
    with h5py.File(f"{format_dataset(dataset_name)}-{format_millions(train_size)}-{X.shape[1]}-euclidean.hdf5", 'w') as f:
        f.create_dataset("train", data=X)
        f.create_dataset("test", data=Y)
        f.create_dataset("neighbors", data=indices)
        f.create_dataset("distances", data=distances)


if __name__ == "__main__":
    typer.run(main)
