import torch
import numpy as np
from loader.datasets import augmentations
import functools

def aug_func(image, preprocess, all_ops, mixture_width, mixture_depth, aug_severity):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(
    np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, all_ops=False, mixture_width=3,
               mixture_depth=-1, aug_severity=3, no_jsd=False, image_size=32):
    # print('using augmix dataset with jsd: {}'.format(not no_jsd))
    augmentations.IMAGE_SIZE = image_size
    # exit()
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.aug = functools.partial(aug_func, all_ops=all_ops, mixture_width=mixture_width,
                                 mixture_depth=mixture_depth, aug_severity=aug_severity)

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return self.aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), self.aug(x, self.preprocess),
                  self.aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)