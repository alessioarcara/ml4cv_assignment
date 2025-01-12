from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class StreetHazards(Dataset):
    def __init__(
            self, 
            root_dir: Path,
            subset: str = "training",
            transforms=None
        ) -> None:
        """
        Args:
            root_dir: Path to the directory containing the image and masks.
            subset: Subfolder name ('training' or 'validation') specifying the dataset split.
            transforms: Transformations to apply the images and masks.
        """
        self.classes = [
            "unlabeled", 
            "building", 
            "fence", 
            "other", 
            "pedestrian", 
            "pole", 
            "road line", 
            "road", 
            "sidewalk", 
            "vegetation", 
            "car", 
            "wall", 
            "traffic sign", 
            "anomaly"
        ]

        images_dir = root_dir / "images" / subset
        masks_dir = root_dir / "annotations" / subset

        self.imgs = sorted(images_dir.glob("*.png"))
        self.masks = sorted(masks_dir.glob("*.png"))

        self.transforms = transforms 

        if len(self.imgs) - len(self.masks) != 0:
            raise AssertionError(
                f"Labels and Images differs in size {len(self.imgs) - len(self.masks)}."
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path_img = self.imgs[idx]
        path_mask = self.masks[idx]

        img = Image.open(path_img).convert("RGB")
        mask = Image.open(path_mask)

        if self.transforms:
            img, mask = self.transforms(img, mask)
        else:
            img = F.to_tensor(img)

        return img, mask, path_mask 
    

if __name__ == "__main__":
    current_dir = Path(__file__).parent  # Directory dello script
    dataset_path = current_dir / "datasets/train"

    dataset = StreetHazards(
        root_dir=dataset_path,
        subset="training/t1-3/"
    )

    img, mask = dataset[0]
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)