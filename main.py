from datasets import load_dataset
from sklearn.neighbors import BallTree
import h5py
import typer

def bytes_to_gb(bytes_value):
    gb_value = bytes_value / (1024 ** 3)
    return gb_value

def format_millions(number):
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M".replace(".0M", "M")
    return str(number)

def format_dataset(name):
    return name.replace("/", "_")


def main(dataset_name:str, train_size: int = 1_000_000, test_size: int = 10_000, k: int = 100, metric: str = "euclidean"):
    assert k < train_size, "k must be less than or equal to the number of training points"

    ds = load_dataset(dataset_name, split=f"train[:{train_size}]", columns=["emb"], num_proc=8)
    ds = ds.with_format("np")
    
    X = ds["emb"]

    kdt = BallTree(X, leaf_size=30, metric=metric)

    distances, indices = kdt.query(X, k=k, return_distance=True)
    
    with h5py.File(f"{format_dataset(dataset_name)}-{format_millions(train_size)}-{X.shape[1]}-{metric}.hdf5", 'w') as f:
        f.create_dataset("train", data=X)
        f.create_dataset("test", data=X[:test_size])
        f.create_dataset("neighbors", data=indices[:test_size])
        f.create_dataset("distances", data=distances[:test_size])


if __name__ == "__main__":
    typer.run(main)
