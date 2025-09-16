from datasets import load_dataset

dataset = load_dataset(
    "/home/algroup/cyq/CP-PLL/cifar10-lt/cifar10-lt.py",
    name="r-200",
    cache_dir="/home/algroup/cyq/CP-PLL/data/cifar-10-lt-200"
)

print(dataset)
