import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from enum import StrEnum


class ShapeMismatchError(Exception):
    pass


class AxisNames(StrEnum):
    X = "x"
    Y = "y"


class DiagramTypes(StrEnum):
    Violin = "violin"
    Hist = "hist"
    Boxplot = "boxplot"


def vizual_axis(
    data: np.ndarray,
    axis: plt.Axes,
    diagram_type: DiagramTypes
) -> None:
    if (diagram_type == DiagramTypes.Boxplot):
        axis.boxplot(
            data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue"),
            medianprops=dict(color="k"),
        )
        axis.set_yticks([])

    elif (diagram_type == DiagramTypes.Violin):
        violin_parts = axis.violinplot(
            data,
            vert=False,
            showmedians=True,
        )

        for body in violin_parts["bodies"]:
            body.set_facecolor("cornflowerblue")
            body.set_edgecolor("blue")

        for part in violin_parts:
            if part == "bodies":
                continue

            violin_parts[part].set_edgecolor("cornflowerblue")

        axis.set_yticks([])

    elif (diagram_type == DiagramTypes.Hist):
        axis.hist(
            data,
            bins=50,
            color="cornflowerblue",
            edgecolor="blue",
            density=True,
        )

    else:
        raise TypeError("Invalid diagram_type")


def visualize_distribution(
    points: np.ndarray,
    diagram_type: Union[DiagramTypes, list[DiagramTypes]],
    diagram_axis: Union[AxisNames, list[AxisNames]],
    path_to_save: str = "",
) -> None:
    plt.style.use("ggplot")

    n = points.shape[0]
    if points.shape != (n, 2):
        raise ShapeMismatchError("Invalid points")

    types = diagram_type if isinstance(diagram_type, list) else [diagram_type]
    coords = diagram_axis if isinstance(diagram_axis, list) else [diagram_axis]

    figure, axs = plt.subplots(
        len(types),
        len(coords),
        figsize=(5 * len(coords), 3 * len(types))
    )

    axes = [axs] if isinstance(axs, plt.Axes) else axs.flatten()
    data = {
        AxisNames.X: points[:, 0],
        AxisNames.Y: points[:, 1]
    }

    for i, type in enumerate(types):
        for j, coord in enumerate(coords):
            vizual_axis(data[coord], axes[len(coords) * i + j], type)
            if i == 0:
                axes[len(coords) * i + j].set_title("X" if coord == AxisNames.X else "Y")

    if (path_to_save):
        plt.savefig(path_to_save)

    plt.show()
