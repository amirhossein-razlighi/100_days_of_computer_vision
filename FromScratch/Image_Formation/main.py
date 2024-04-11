import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_camera_matrix(focal_length: float, sensor_size: tuple, image_size: tuple):
    fx = focal_length * image_size[0] / sensor_size[0]
    fy = focal_length * image_size[1] / sensor_size[1]
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_projection_matrix(
    camera_matrix: np.array, rotation_matrix: np.array, translation_vector: np.array
):
    return np.dot(
        camera_matrix, np.hstack((rotation_matrix, translation_vector[:, np.newaxis]))
    )


if __name__ == "__main__":
    f, cx, cy, noise_std = 56.25, 320, 240, 1
    image_res = (640, 480)
    cam_pos = [[0, 0, 0], [-2, -2, 0], [2, 2, 0], [-2, 2, 0], [2, -2, 0]]  # Unit: [m]
    cam_ori = [
        [0, 0, 0],
        [-15, 15, 0],
        [15, -15, 0],
        [15, 15, 0],
        [-15, -15, 0],
    ]  # Unit: [deg]

    X = np.loadtxt("./box.xyz")  # Load 3D points

    fig = go.Figure(
        data=[go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers")]
    )

    # Add camera poses and their viewing direction to the 3D scatter plot
    for i, (pos, ori) in enumerate(zip(cam_pos, cam_ori)):
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        )

        # Calculate camera viewing direction
        R_matrix = R.from_euler("zyx", ori[::-1], degrees=True).as_matrix()
        direction = np.dot(R_matrix, np.array([0, 0, 1]))

        # Add camera viewing direction
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0], pos[0] + direction[0]],
                y=[pos[1], pos[1] + direction[1]],
                z=[pos[2], pos[2] + direction[2]],
                mode="lines",
                line=dict(color="blue", width=5),
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Scene and Camera Poses",
    )
    fig.show()

    K = get_camera_matrix(f, (36, 27), image_res)
    subs = make_subplots(
        rows=len(cam_pos),
        cols=3,
        column_widths=[0.4, 0.4, 0.2],
    )
    for i, (pos, ori) in enumerate(zip(cam_pos, cam_ori)):
        R_matrix = R.from_euler("xyz", ori, degrees=True).as_matrix()
        R_ = R_matrix.T
        T_vector = -np.dot(R_matrix.T, pos)
        P = get_projection_matrix(K, R_, T_vector)

        x = P @ np.vstack((X.T, np.ones(len(X))))
        x /= x[-1]

        x[0:2, :] += np.random.normal(scale=noise_std, size=(2, len(X)))

        img = np.zeros(image_res[::-1], dtype=np.uint8)
        for c in range(x.shape[1]):
            if 0 <= x[1, c] < image_res[1] and 0 <= x[0, c] < image_res[0]:
                img[int(x[1, c]), int(x[0, c])] = 255

        fig_2d = go.Figure(
            data=[go.Heatmap(z=img, colorscale="gray", zmin=0, zmax=255)]
        )
        subs.add_trace(fig_2d.data[0], row=i + 1, col=2)
        subs.add_trace(
            go.Scatter(x=x[0], y=x[1], mode="markers", marker=dict(size=5)),
            row=i + 1,
            col=1,
        )

    subs.update_layout(title_text="2D Image Formation")
    subs.show()
