# Validation record

Windows CPU with Python 3.10.20, PyTorch 2.7.1 and torchvision 0.22.1.

- Loaded the official Fashion-MNIST training and test sets and the lab's `resnet18_fash.pth` and `resnet18_fash.npy`.
- Ran `Main_embedding.py`, `Model_attack.py` and `Main_recovery.py` in order with four test images and one trial at the script's 0.1 tampering rate. Verified the marked checkpoint, recovery arrays, tampered checkpoint, tampering mask and recovery log.
- Ran `Score_cal.py` for one batch of two Fashion-MNIST training images. Verified that the saved ranking contained each of the 40,000 parameter indices exactly once.

The tests cover the default Fashion-MNIST and ResNet-18 entry points. Full test-set accuracy, other backbone and dataset combinations, and paper benchmark reproduction remain outside these checks.
