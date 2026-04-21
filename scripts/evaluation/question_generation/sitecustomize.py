# sitecustomize.py
# Auto-imported by Python if it's on sys.path.
# We patch torchvision.transforms.InterpolationMode for older torchvision versions.

try:
    import torchvision.transforms as T
    if not hasattr(T, "InterpolationMode"):
        class InterpolationMode:
            NEAREST = 0
            BILINEAR = 2
            BICUBIC = 3
            LANCZOS = 1
            HAMMING = 4
            BOX = 5
        T.InterpolationMode = InterpolationMode
except Exception:
    # If torchvision isn't importable, do nothing
    pass