import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from knn.realization import KNearestNeighbors, WeightedKNearestNeighbors
from functools import partial
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_sphere(
        center: np.ndarray, 
        radius: float,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        resolution: float = 20,
        ):
    u, v = np.mgrid[0:(2 * np.pi):resolution * 1j, 0:(np.pi):resolution * 1j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]

    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    z = np.clip(z, z_min, z_max)
    return x, y, z

class AnimationKNN:
    def __init__(self, x_test: np.ndarray):
        self.x_test = x_test

    def create_animation(
        self,
        knn: WeightedKNearestNeighbors | KNearestNeighbors,
        true_targets: np.ndarray,
        path_to_save: str = "",
    ) -> FuncAnimation:
        plt.style.use("ggplot")

        x_train = knn._x_train
        y_train = knn._y_train
        unique_labels = np.unique(y_train)
        first_label = unique_labels[0]
        second_label = unique_labels[1]
        first_indices = np.where(y_train == first_label)
        second_indices = np.where(y_train == second_label)
        first_points = x_train[first_indices]
        second_points = x_train[second_indices]
        first_color = "b"
        second_color = "r"

        if self.x_test.shape[1] == 2:

            figure, axis = plt.subplots(figsize=(10, 8))
            axis: plt.Axes = axis
            axis.set_aspect('equal', adjustable='box')
            axis.scatter(first_points[:, 0], first_points[:, 1], c=first_color)
            axis.scatter(second_points[:, 0], second_points[:, 1], c=second_color)

            y_test = knn.predict(self.x_test)

            circle = None

            eps_x = (x_train[:, 0].max() - x_train[:, 0].min()) / 10
            eps_y = (x_train[:, 1].max() - x_train[:, 1].min()) / 10
            x_min, x_max = x_train[:, 0].min() - eps_x, x_train[:, 0].max() + eps_x
            y_min, y_max = x_train[:, 1].min() - eps_y, x_train[:, 1].max() + eps_y
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)

            def update_frame(
                frame_id: int,
                *,
                axis: plt.Axes
            ) -> list[plt.Circle]:
                nonlocal circle

                if circle is not None:
                    circle.remove()

                current_point = self.x_test[frame_id, :]
                x = np.round(current_point[0], 2)
                y = np.round(current_point[1], 2)

                predict_label = y_test[frame_id]
                true_label = true_targets[frame_id]

                color = first_color if (predict_label == first_label) else second_color
                axis.scatter(x, y, c=color)

                distances = knn._calc_distances(np.array([current_point]), x_train)  # Находим Р
                radius = np.max(np.sort(distances, axis=-1)[:, :knn._n_neighbors])

                circle = plt.Circle(
                    (x, y),
                    radius=radius,
                    alpha=0.5,
                    color='green',
                    fill=True,
                    clip_on=True
                )
                axis.add_patch(circle)

                title = f"({x}, {y}) - {"Верно" if true_label == predict_label else "Неверно"}"

                axis.set_title(title)

                return [circle]

            animation = FuncAnimation(
                figure,
                partial(update_frame, axis=axis),
                frames=len(y_test),
                interval=200,
                blit=False
            )

            if (path_to_save):
                animation.save(path_to_save, writer="pillow", fps=24)

            plt.show()

            return animation

        else:

            figure, axis = plt.subplots(figsize=(9, 9))
            axis: plt.Axes = figure.add_subplot(projection="3d")

            y_test = knn.predict(self.x_test)

            drawn_artists = []

            eps_x = (x_train[:, 0].max() - x_train[:, 0].min()) / 10
            eps_y = (x_train[:, 1].max() - x_train[:, 1].min()) / 10
            eps_z = (x_train[:, 2].max() - x_train[:, 2].min()) / 10

            x_min, x_max = x_train[:, 0].min() - eps_x, x_train[:, 0].max() + eps_x
            y_min, y_max = x_train[:, 1].min() - eps_y, x_train[:, 1].max() + eps_y
            z_min, z_max = x_train[:, 2].min() - eps_z, x_train[:, 2].max() + eps_z

            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_ylim(z_min, z_max)

            def update_frame(
                frame_id: int,
                *,
                axis: plt.Axes
            ) -> list[plt.Circle]:
                nonlocal drawn_artists

                axis.clear()

                axis.scatter3D(
                    first_points[:, 0], 
                    first_points[:, 1], 
                    first_points[:, 2], 
                    c=first_color
                    )
                axis.scatter3D(
                    second_points[:, 0], 
                    second_points[:, 1], 
                    second_points[:, 2], 
                    c=second_color
                    )
                axis.set_aspect('equal', adjustable='box')
                axis.set_xlim(x_min, x_max)
                axis.set_ylim(y_min, y_max)
                axis.set_zlim(z_min, z_max)

                current_point = self.x_test[frame_id, :]
                x = np.round(current_point[0], 2)
                y = np.round(current_point[1], 2)
                z = np.round(current_point[2], 2)

                predict_label = y_test[frame_id]
                true_label = true_targets[frame_id]

                color = first_color if (predict_label == first_label) else second_color
                axis.scatter3D(
                    x,
                    y,
                    z,
                    c=color
                )

                distances = knn._calc_distances(np.array([current_point]), x_train)  # Находим Р
                radius = np.max(np.sort(distances, axis=-1)[:, :knn._n_neighbors])

                x_sph, y_sph, z_sph = create_sphere(
                    current_point,
                    radius,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max
                    )
                sphere = (x_sph, y_sph, z_sph)

                axis.plot_surface(x_sph, y_sph, z_sph, color="green", alpha=0.5)
                drawn_artists.append(sphere)

                title = f"({x}, {y}, {z}) - {"Верно" if true_label == predict_label else "Неверно"}"

                axis.set_title(title)

                return drawn_artists

            animation = FuncAnimation(
                figure,
                partial(update_frame, axis=axis),
                frames=len(y_test),
                interval=200,
                blit=False
            )

            if (path_to_save):
                animation.save(path_to_save, writer="pillow", fps=24)

            plt.show()

            return animation