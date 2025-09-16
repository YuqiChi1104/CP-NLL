# run_dataset.py
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Load CIFAR10-LT dataset")
    parser.add_argument("--script_path", type=str, required=True,
                        help="Path to the dataset script (e.g., cifar10-lt.py)")
    parser.add_argument("--name", type=str, default="r-200",
                        help="Dataset name or configuration (e.g., r-200)")
    parser.add_argument("--cache_dir", type=str, default="./data/cifar-10-lt-200",
                        help="Cache directory to store dataset")

    args = parser.parse_args()

    dataset = load_dataset(
        args.script_path,
        name=args.name,
        cache_dir=args.cache_dir
    )

    print(dataset)

if __name__ == "__main__":
    main()

