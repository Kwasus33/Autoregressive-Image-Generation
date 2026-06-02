from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm


def main():
    tiny_imagenet_split_names = get_dataset_split_names("zh-plus/tiny-imagenet")
    print(tiny_imagenet_split_names)
    tiny_imagenet = load_dataset("zh-plus/tiny-imagenet")
    


if __name__ == "__main__":
    main()
