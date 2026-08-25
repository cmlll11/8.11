"""Allow torchvision to read the intentionally rebatched experiment data."""

from __future__ import annotations

import os

from torchvision.datasets import CIFAR10


_original_check_integrity = CIFAR10._check_integrity


def _check_integrity(self) -> bool:
    root = os.path.abspath(str(self.root))
    if "hard_sample_gap" not in root:
        return _original_check_integrity(self)
    required = [filename for filename, _ in self.train_list + self.test_list]
    return all(os.path.isfile(os.path.join(self.root, self.base_folder, name)) for name in required)


CIFAR10._check_integrity = _check_integrity
