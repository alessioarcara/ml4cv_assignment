from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2 as cv

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

        self.imgs = sorted(images_dir.rglob("*.png"))
        self.masks = sorted(masks_dir.rglob("*.png"))

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

        img = cv.cvtColor(cv.imread(path_img, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        mask = cv.imread(path_mask, cv.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask
    
    def get_classes_as_dict(self):
        return { idx: class_name for idx, class_name in enumerate(self.classes) }
    

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