import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from prnet.prnet import PRNFeatures
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional


@dataclass
class PreprocessingConfig:
    image_size: int=256
    depth_map_size: int = 64
    spoof_depth_map_value: float = 0.5
    differential_matrix: torch.Tensor = torch.tensor(
        [
            [1, -1, 0, 0, 0],
            [1, 0, -1, 0, 0],
            [1, 0, 0, -1, 0],
            [0, 1, 0, 0, -1],
            [0, 0, 1, 0, -1],
            [0, 0, 0, 1, -1]
            ], dtype=torch.float32
            )
    
class ImageProcessor:
    @staticmethod
    def normalize(image: npt.NDArray) -> npt.NDArray:
        if image.max() == image.min():
            return np.zeros_like(image)
        return (image - image.min()) / (image.max() - image.min())
    
    @staticmethod
    def resize(image: npt.NDArray, size: Tuple[int, int]) -> npt.NDArray:
        return cv2.resize(image, size)
    
    @staticmethod
    def bgr_to_rgb(image: npt.NDArray) -> npt.NDArray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
class Preprocessing:
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.features = PRNFeatures()
        self.image_processor = ImageProcessor()
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config.image_size, self.config.image_size)
            ),
            transforms.ToTensor()
        ])

    def crop_face(self,
                  image: npt.NDArray,
                  align: bool = True,
                  return_shape: int = PreprocessingConfig.image_size) -> Optional[npt.NDArray]:
        try:
            image = ImageProcessor.bgr_to_rgb(image)
            if align:
                image = self.features.face_alignment(image)
            crop = self.features.face_crop(image)
            if crop is None or crop.size == 0:
                return
            crop = self.image_processor.resize(crop,
                                               (return_shape,
                                                return_shape)
                                                )
            return crop
        except Exception as e:
            print(f"error crop face: {e}")
            return
        
    def depth_maps(self, image: npt.NDArray) -> npt.NDArray:
        depth_map = self.features.get_depth_map(image, shape=self.config.image_size)
        depth_map = self.image_processor.resize(depth_map, (self.config.depth_map_size, self.config.depth_map_size))
        return self.normalize(depth_map)
    
    def differential_norm(self, images: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            images = torch.stack(images)
        return torch.tensordot(self.config.differential_matrix, images, dims=1)
    
    @staticmethod
    def stack_image(images: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(images)
    
    @staticmethod
    def spoof_depth_map():
        return np.full((64, 64), 0.5, dtype=np.float32)

    def normalize(self, image: npt.NDArray) -> npt.NDArray:
        return self.image_processor.normalize(image)