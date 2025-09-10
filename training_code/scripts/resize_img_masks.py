import os
import shutil


def copy_images_masks_with_numbers(src_images: str, src_masks: str, dst_images: str, dst_masks: str,
                                   numbers: list[int]):
    """
    Copy images and masks if their filenames contain any of the given numbers.

    Args:
        src_images: Path to images folder
        src_masks: Path to masks folder
        dst_images: Destination folder for copied images
        dst_masks: Destination folder for copied masks
        numbers: List of numbers to search for in filenames (as substrings)
    """
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_masks, exist_ok=True)

    # Convert numbers to strings for substring matching
    number_strs = [str(n) for n in numbers]

    def matches_number(filename: str) -> bool:
        return any(num in filename for num in number_strs)

    # Copy images
    for f in os.listdir(src_images):
        if f.lower().endswith((".jpg", ".png", ".jpeg")) and matches_number(f):
            shutil.copy2(os.path.join(src_images, f), os.path.join(dst_images, f))

    # Copy masks
    for f in os.listdir(src_masks):
        if f.lower().endswith((".jpg", ".png", ".jpeg")) and matches_number(f):
            shutil.copy2(os.path.join(src_masks, f), os.path.join(dst_masks, f))

    print(f"Copied matching images to {dst_images} and masks to {dst_masks}")


numbers = [
    159,171,389,2105,5147,5657,2951,2178,2190,905,1904,2139,3212,3361,5978,
    14,15,229,231,354,2066,2432,4165,2926,3476,5012,492,553,566,1640,1650,
    79,237,238,240,257,264,266,267,467,745,969,971,164,232,479,483,695,930,
    1266,1586,1588,1665,1795,77,26,49,63,68,900,702,1147,3473,230,265,351,
    355,357,384,400,440,4140,5798,4803,118,552,554,1084,1638,1651,215,254,
    789,454,688,700,709,1593,1673,1723,1813,35,36,38,93,1309,804,912,1735,
    211,277,319,703,958,1903,1906,2098,2100,2101,2145,2259,2345,3261,3266,
    3270,150,153,421,431,444,446,452,480,491,508,2180,2756,2785,4154,4790,
    4835,4986,5153,5777,2744,2771,2846,2847,2848,3168,3232,3286,4226,4478,
    5256,5330,5334,5979,5980,6313,6628,1086,1687,78,260,104,1064,2186,1062,
    1441,1454,327,201
]

copy_images_masks_with_numbers(
    src_images=r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_IMAGES_CLASS_4",
    src_masks=r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_MASKS_CLASS_4",
    dst_images=r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_CLASSImages",
    dst_masks=r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\DATASET_V2\DATASET_CLASSMasks",
    numbers=numbers
)
