import torch

checkpoint_path = r"D:\Devendra_Files\segmentation_training\weights\Unetpp\latest_checkpoint.pth"

checkpoint = torch.load(checkpoint_path,map_location="cpu")


def print_recursive(obj, indent=0):
    prefix = " " * indent

    if isinstance(obj, dict):
        print(f"{prefix}DICT ({len(obj)} keys)")
        for k, v in obj.items():
            print(f"{prefix}[{k}]")
            print_recursive(v, indent + 4)

    elif isinstance(obj, list):
        print(f"{prefix}LIST (len={len(obj)})")
        for i, item in enumerate(obj):
            print(f"{prefix}[{i}]")
            print_recursive(item, indent + 4)

    elif isinstance(obj, torch.Tensor):
        print(
            f"{prefix}Tensor "
            f"shape={tuple(obj.shape)} "
            f"dtype={obj.dtype}"
        )

        # print actual values for small tensors
        if obj.numel() <= 20:
            print(f"{prefix}{obj}")

    else:
        print(f"{prefix}{type(obj).__name__}: {obj}")


print_recursive(checkpoint)
