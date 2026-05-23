"""
proposed_method/dataset.py
--------------------------
Provides ModelNet40Dataset as expected by evaluate_all_extended.py.

The evaluator calls:
    ModelNet40Dataset(root=args.data_root, split='test', npoints=1024)

ModelNet40H5 in train.py uses:
    ModelNet40H5(data_root=..., mode=..., n_points=...)

This shim maps the two interfaces together.
"""

from .train import ModelNet40H5


class ModelNet40Dataset(ModelNet40H5):
    """
    Thin wrapper around ModelNet40H5 that accepts the keyword arguments
    used by evaluate_all_extended.py:
        root    -> data_root
        split   -> mode  ('test' / 'train')
        npoints -> n_points
    """

    def __init__(
        self,
        root:    str,
        split:   str = "test",
        npoints: int = 1024,
        augment: bool = False,
    ) -> None:
        # augment defaults to False for evaluation splits
        super().__init__(
            data_root=root,
            mode=split,
            n_points=npoints,
            augment=augment,
        )


__all__ = ["ModelNet40Dataset"]
