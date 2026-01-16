from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Tuple, Callable, Optional, List
from torch import Tensor
import pandas as pd
from PIL import Image
import os


class BaseDataset(Dataset):
    """Base class for WaterbirdDataset and SpeciesDataset."""

    def __init__(self, annotation_file: str, image_dir: str,
                 transform: Optional[Callable[[Image.Image], Tensor]] = None,
                 train: bool = True):
        """
        Initialize the dataset.

        Args:
            annotation_file (str): Path to the annotation CSV file.
            image_dir (str): Directory containing the images.
            transform (callable, optional): Transform to apply to the images.
            train (bool): Whether to use the training set (True) or test set (False).
        """
        self.transform = transform or ToTensor()
        self.image_dir = image_dir

        self.annotations = pd.read_csv(annotation_file)
        self.annotations = self.annotations[self.annotations["split"] == (0 if train else 1)]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)

    def _load_image(self, file_path: str) -> Image.Image:
        """Load an image from file_path and convert it to RGB."""
        return Image.open(os.path.join(self.image_dir, file_path)).convert('RGB')


class WaterbirdDataset(BaseDataset):
    """Dataset for waterbird/landbird classification."""

    def __init__(self, annotation_file: str, image_dir: str,
                 transform: Optional[Callable[[Image.Image], Tensor]] = None,
                 train: bool = True):
        super().__init__(annotation_file, image_dir, transform, train)
        self.label_names = ["waterbird", "landbird"]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed PIL Image and
                   label is an integer (0 for waterbird, 1 for landbird).
        """
        row = self.annotations.iloc[idx]
        image = self._load_image(row["img_filename"])
        label = row["birdtype"]
        return self.transform(image), label


class SpeciesDataset(BaseDataset):
    """Dataset for bird species classification."""

    def __init__(self, annotation_file: str, image_dir: str,
                 transform: Optional[Callable[[Image.Image], Tensor]] = None,
                 train: bool = True):
        super().__init__(annotation_file, image_dir, transform, train)
        self.annotations["species"], self.species = pd.factorize(self.annotations["species"])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed PIL Image and
                   label is an integer representing the species.
        """
        row = self.annotations.iloc[idx]
        image = self._load_image(row["img_filename"])
        label = row["species"]
        return self.transform(image), label

    @property
    def num_classes(self) -> int:
        """Return the number of unique species in the dataset."""
        return len(self.species)

    def get_species_name(self, species_idx: int) -> str:
        """
        Get the species name for a given species index.

        Args:
            species_idx (int): The index of the species.

        Returns:
            str: The name of the species.
        """
        return self.species[species_idx]