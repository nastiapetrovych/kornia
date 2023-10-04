from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils import _extract_device_dtype


class RainGenerator(RandomGeneratorBase):
    def __init__(
        self, number_of_drops: tuple[int, int], drop_height: tuple[int, int], drop_width: tuple[int, int]
    ) -> None:
        super().__init__()
        self.number_of_drops = number_of_drops
        self.drop_height = drop_height
        self.drop_width = drop_width

    def __repr__(self) -> str:
        repr = f"number_of_drops={self.number_of_drops}, drop_height={self.drop_height}, drop_width={self.drop_width}"
        return repr

    def draw_line(image: torch.Tensor, params_val: dict[str, Tensor], streak_intensity=0.02) -> torch.Tensor:
    """ 
    Draws rain streaks on the image based on the provided parameters
    :param 'params_val': dictionary containing rain parameters:
          'number_of_drops_factor': number of rain streaks/drops.
          'coordinates_factor': starting coordinates for each drop.
          'drop_height_factor': height for each drop/streak.
          'drop_width_factor': width for each drop/streak (usually 1 for streak).
    :param 'streak_intensity': float 
    :returns: torch.Tensor
    """
    coords = params_val['coordinates_factor']
    heights = params_val['drop_height_factor']
    widths = params_val['drop_width_factor']
    B, C, H, W = image.shape
    for i in range(B):  # loop over each image in the batch
        num_drops = params_val['number_of_drops_factor'][i].item()
        for j in range(num_drops):
            # calculate starting and ending coordinates for the streak
            y_start = int(coords[i, j, 0].item() * H)
            x_start = int(coords[i, j, 1].item() * W)
            y_end = y_start + heights[i].item()
            x_end = x_start + widths[i].item()
            image[i, :, y_start:y_end, x_start:x_end] += streak_intensity  # draw the streak on the image
    image = torch.clamp(image, 0, 1)  # clamp values to be in [0, 1] range
    return image

    
    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        number_of_drops = _range_bound(
            self.number_of_drops,
            'number_of_drops',
            center=self.number_of_drops[0] / 2 + self.number_of_drops[1] / 2,
            bounds=(self.number_of_drops[0], self.number_of_drops[1] + 1),
        ).to(device)
        drop_height = _range_bound(
            self.drop_height,
            'drop_height',
            center=self.drop_height[0] / 2 + self.drop_height[1] / 2,
            bounds=(self.drop_height[0], self.drop_height[1] + 1),
        ).to(device)
        drop_width = _range_bound(
            self.drop_width,
            'drop_width',
            center=self.drop_width[0] / 2 + self.drop_width[1] / 2,
            bounds=(self.drop_width[0], self.drop_width[1] + 1),
        ).to(device)

        drop_coordinates = _range_bound((0, 1), 'drops_coordinate', center=0.5, bounds=(0, 1)).to(
            device=device, dtype=dtype
        )
        self.number_of_drops_sampler = UniformDistribution(number_of_drops[0], number_of_drops[1], validate_args=False)
        self.drop_height_sampler = UniformDistribution(drop_height[0], drop_height[1], validate_args=False)
        self.drop_width_sampler = UniformDistribution(drop_width[0], drop_width[1], validate_args=False)
        self.coordinates_sampler = UniformDistribution(drop_coordinates[0], drop_coordinates[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.drop_width, self.drop_height, self.number_of_drops])
        # self.ksize_factor.expand((batch_size, -1))
        number_of_drops_factor = _adapted_rsampling((batch_size,), self.number_of_drops_sampler).to(
            device=_device, dtype=torch.long
        )
        drop_height_factor = _adapted_rsampling((batch_size,), self.drop_height_sampler, same_on_batch).to(
            device=_device, dtype=torch.long
        )
        drop_width_factor = _adapted_rsampling((batch_size,), self.drop_width_sampler, same_on_batch).to(
            device=_device, dtype=torch.long
        )
        coordinates_factor = _adapted_rsampling(
            (batch_size, int(number_of_drops_factor.max().item()), 2),
            self.coordinates_sampler,
            same_on_batch=same_on_batch,
        ).to(device=_device)
        return {
            'number_of_drops_factor': number_of_drops_factor,
            'coordinates_factor': coordinates_factor,
            'drop_height_factor': drop_height_factor,
            'drop_width_factor': drop_width_factor,
        }
