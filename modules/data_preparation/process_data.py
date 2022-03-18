import cv2
import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset parameters
N_KEYPOINTS = 21
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 128
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}


def project_3d_points_to_2d(xyz, K):
    """
    Projects 3D coordinates into image space.
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, 2:]


def vector_to_heatmaps(keypoints):
    """
    Creates 2D heatmaps from keypoint locations for a single image.
    """
    if keypoints.shape != (N_KEYPOINTS, 2):
        raise Exception("There is a problem with the size of the key points.")

    heatmaps = np.zeros([N_KEYPOINTS, MODEL_IMG_SIZE, MODEL_IMG_SIZE])
    scale_factor = MODEL_IMG_SIZE / RAW_IMG_SIZE
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * scale_factor), int(y * scale_factor)
        if (0 <= x < MODEL_IMG_SIZE) and (0 <= y < MODEL_IMG_SIZE):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred


def show_dataset_sample(dataset, n_samples=12):
    """
    Visualize data n_samples from a dataset.
    """
    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[15, n_rows * 4])

    ids = np.random.choice(dataset.__len__(), n_samples, replace=False)
    for plot_id, (id) in enumerate(ids, 1):
        sample = dataset.__getitem__(id)

        image = sample["image_raw"].numpy()
        image = np.moveaxis(image, 0, -1)
        keypoints = sample["keypoints"].numpy()

        plt.subplot(n_rows, n_cols, plot_id)
        plt.imshow(image)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=0.5)
        for _, params in COLORMAP.items():
            plt.plot(keypoints[params["ids"], 0], keypoints[params["ids"], 1], params["color"])
            plt.title(sample["image_name"])
    plt.tight_layout()
    plt.show()


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Augmented images are not used, only raw - first 32,560 images
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, device, data_dir, set_type="train"):
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise Exception(f"The data directory for the database {data_dir} doesn't exist.")

        self.device = device
        # Get the names of the images
        self.image_dir = data_dir.joinpath("training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))
        # Get the intrinsics matrix per image
        fn_K_matrix = data_dir.joinpath("training_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))
        # Get the ground truth positions
        ground_truth =data_dir.joinpath("training_xyz.json")
        with open(ground_truth, "r") as f:
            self.ground_truth = np.array(json.load(f))

        # TODO redo this and change the values of mean and std
        if set_type == "train":
            n_start = 0
            n_end = 26000
        elif set_type == "val":
            n_start = 26000
            n_end = 31000
        else:
            n_start = 31000
            n_end = len(self.ground_truth)

        self.image_names = self.image_names[n_start:n_end]
        self.K_matrix = self.K_matrix[n_start:n_end]
        self.ground_truth = self.ground_truth[n_start:n_end]

        # Transformations to do in the images when loaded
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_raw = Image.open(str(self.image_dir.joinpath(image_name)))
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)

        keypoints = project_3d_points_to_2d(self.ground_truth[idx], self.K_matrix[idx])
        heatmaps = vector_to_heatmaps(keypoints)
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
        }
