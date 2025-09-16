from datasets import Dataset

# 读取 arrow 文件
dataset = Dataset.from_file("/home/algroup/cyq/cp_pll/data/cifar-10-lt-200/cifar10-train.arrow")

# 打乱（每次运行都会不一样）
shuffled_dataset = dataset.shuffle(seed=42)  # 可以去掉 seed，让每次随机都不一样


shuffled_dataset._data.table.to_batches()
shuffled_dataset.save_to_disk("/home/algroup/cyq/cp_pll/data/cifar-10-lt-200/cifar10-train_s.arrow")
