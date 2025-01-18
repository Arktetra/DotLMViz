from torch.utils.data import DataLoader
from typing import Optional

import DoTLMViz.metadata.shared as metadata


class DataModule:
    """Base class for data modules."""

    def __init__(self, batch_size: int = 32, num_workers: int = 0, accelerators: int = None) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = accelerators
        self.shuffle = True

    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIRNAME

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
