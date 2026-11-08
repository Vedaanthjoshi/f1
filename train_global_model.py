"""
train_global_model.py — Training Script Facade (Delegates to ml/trainer.py)
===========================================================================
Maintained for backward compatibility. Training logic has been
componentized into the `ml/trainer.py` module.
"""

from ml.trainer import train_global_model


def train_global():
    train_global_model()


if __name__ == "__main__":
    train_global()
