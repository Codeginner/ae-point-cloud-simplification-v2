import torch
import numpy as np
import open3d as o3d

from .model import PointCloudSimplifier
from .train import PointCloudDataset


# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------

model = PointCloudSimplifier(
    M=512
).to(device)


# ----------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------

checkpoint = torch.load(
    "/kaggle/working/ae-point-cloud-simplification/checkpoints/best.pth", # change with your work dir
    map_location=device
)

print(checkpoint.keys())


# ----------------------------------------------------------
# Load weights
# ----------------------------------------------------------

# Kalau checkpoint punya key "model"
# model.load_state_dict(checkpoint["model"])

# Kalau error, ganti jadi:
model.load_state_dict(checkpoint)

model.eval()


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

dataset = PointCloudDataset(
    data_root="./data",
    mode="test",
    n_points=1024,
    augment=False
)


# ----------------------------------------------------------
# Ambil 1 sample
# ----------------------------------------------------------

#P, label = dataset[0]

# ----------------------------------------------------------
# Find one chair sample
# ----------------------------------------------------------

chair_label = 2

chair_idx = None

for i in range(len(dataset)):

    _, label = dataset[i]

    if int(label) == chair_label:

        chair_idx = i
        break


# ----------------------------------------------------------
# Load chair sample
# ----------------------------------------------------------

P, label = dataset[chair_idx]

P = P.unsqueeze(0).to(device)


# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------

with torch.no_grad():

    out = model(P)

    P_original = P[0].cpu().numpy()

    P_simplified = out["P_simplified"][0].cpu().numpy()

    P_recon = out["P_recon"][0].cpu().numpy()


# ----------------------------------------------------------
# Open3D helper
# ----------------------------------------------------------

def to_o3d(points, color):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(
        np.array(color),
        (points.shape[0], 1)
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ----------------------------------------------------------
# Build point clouds
# ----------------------------------------------------------

pcd_original = to_o3d(
    P_original,
    [0.2, 0.4, 1.0]
)

pcd_simplified = to_o3d(
    P_simplified,
    [1.0, 0.2, 0.2]
)

pcd_recon = to_o3d(
    P_recon,
    [0.2, 1.0, 0.2]
)


# ----------------------------------------------------------
# Shift clouds
# ----------------------------------------------------------

pcd_original.translate((-2.5, 0, 0))

pcd_recon.translate((2.5, 0, 0))


# ----------------------------------------------------------
# Visualize
# ----------------------------------------------------------
'''
o3d.visualization.draw_geometries(
    [
        pcd_original,
        pcd_simplified,
        pcd_recon
    ],
    window_name="Original | Simplified | Reconstructed",
    width=1600,
    height=700
)
'''
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Matplotlib visualization
# ----------------------------------------------------------

fig = plt.figure(figsize=(18, 6))


# ----------------------------------------------------------
# Original
# ----------------------------------------------------------

ax1 = fig.add_subplot(131, projection='3d')

ax1.scatter(
    P_original[:, 0],
    P_original[:, 1],
    P_original[:, 2],
    s=1
)

ax1.set_title("Original")


# ----------------------------------------------------------
# Simplified
# ----------------------------------------------------------

ax2 = fig.add_subplot(132, projection='3d')

ax2.scatter(
    P_simplified[:, 0],
    P_simplified[:, 1],
    P_simplified[:, 2],
    s=3
)

ax2.set_title("Simplified")


# ----------------------------------------------------------
# Reconstructed
# ----------------------------------------------------------

ax3 = fig.add_subplot(133, projection='3d')

ax3.scatter(
    P_recon[:, 0],
    P_recon[:, 1],
    P_recon[:, 2],
    s=3
)

ax3.set_title("Reconstructed")


# ----------------------------------------------------------
# Remove axis
# ----------------------------------------------------------

for ax in [ax1, ax2, ax3]:

    ax.set_axis_off()


plt.tight_layout()

plt.savefig(
    "visualization_result.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()