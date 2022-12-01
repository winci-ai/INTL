import random
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms
import torch

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transforms)
    

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

class ImageNetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
    ):
        """Class that applies Imagenet transformations.
        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def MultiCrops(cfg):

    if cfg.multicrop == 1:
        print("=> apply multiple 4 crops: 192, 160, 128, 96")
        cfg.nmb_crops = [1, 1, 1, 1]
        cfg.crops_size = [192, 160, 128, 96]
        cfg.min_scale_crops = [0.2, 0.167, 0.133, 0.1]
        cfg.max_scale_crops = [1.0, 0.833, 0.667, 0.5]
        cfg.gaussian_prob = [0.5, 0.5, 0.5, 0.5]
        cfg.solarization_prob = [0.1, 0.1, 0.1, 0.1]

    if cfg.multicrop == 2:
        print("=> apply multiple 5 crops: 224, 192, 160, 128, 96")
        cfg.nmb_crops = [1, 1, 1, 1, 1]
        cfg.crops_size = [224, 192, 160, 128, 96]
        cfg.min_scale_crops = [0.2, 0.171, 0.143, 0.114, 0.086]
        cfg.max_scale_crops = [1.0, 0.857, 0.714, 0.571, 0.429]
        cfg.gaussian_prob = [0.5, 0.5, 0.5, 0.5, 0.5]
        cfg.solarization_prob = [0.1, 0.1, 0.1, 0.1, 0.1]

    if cfg.multicrop == 3:
        print("=> apply multiple 6 crops: 2 x 224, 192, 160, 128, 96")
        cfg.nmb_crops = [2, 1, 1, 1, 1]
        cfg.crops_size = [224, 192, 160, 128, 96]
        cfg.min_scale_crops = [0.2, 0.171, 0.143, 0.114, 0.086]
        cfg.max_scale_crops = [1.0, 0.857, 0.714, 0.571, 0.429]
        cfg.gaussian_prob = [0.5, 0.5, 0.5, 0.5, 0.5]
        cfg.solarization_prob = [0.1, 0.1, 0.1, 0.1, 0.1]

    return cfg
