import argparse
from datasets import Dataset

def main(args):
    # 读取 arrow 文件
    dataset = Dataset.from_file(args.input)

    # 打乱
    if args.seed is not None:
        shuffled_dataset = dataset.shuffle(seed=args.seed)
    else:
        shuffled_dataset = dataset.shuffle()

    # 保存
    shuffled_dataset.save_to_disk(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="输入 arrow 文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出保存路径 (目录)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (可选，不设置则每次随机都不同)")
    args = parser.parse_args()

    main(args)
