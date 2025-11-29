import embedding_scoring
import hyperbolic_analysis
import pipeline_romance
import enums
import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as mpatches
from typing import Dict, List
from scipy.linalg import eig
from copy import deepcopy


def plot_spectrogram(
    spect_arr: np.array,
    rotate: bool = False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat.

    Plots a spectrogram value array.

    Args:
        spect_arr (np.array): The spectrogram value array to plot.
        rotate (bool, optional): Whether or not the spectrogram surface is rotated
                                 90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(
        0,
        pipeline_romance._get_TIME_UPPER_BOUND(),
        pipeline_romance._get_TIME_INTERVAL(),
    )
    Y = np.arange(
        0,
        pipeline_romance._get_FREQ_UPPER_BOUND(),
        pipeline_romance._get_FREQ_INTERVAL(),
    )
    X, Y = np.meshgrid(X, Y)

    spect_arr = spect_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            Y,
            X,
            spect_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-80,
            vmax=35,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Time (standardized)")
        ax.invert_yaxis()
        plt.title("Intensity (dB)")
    else:
        surf = ax.plot_surface(
            X,
            Y,
            spect_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-80,
            vmax=40,
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Frequency (Hz)")
        plt.title("Intensity (dB)")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(
        surf,
        shrink=0.5,
        aspect=5,
    )

    plt.show()


def plot_time_covariance(
    cov_arr: np.array,
    rotate=False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat.

    Plots a time covariance array.

    Args:
        cov_arr (np.array): The time covariance array to plot.
        rotate (bool, optional): Whether or not the covariance surface is rotated
                                 90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(
        0,
        pipeline_romance._get_TIME_UPPER_BOUND(),
        pipeline_romance._get_TIME_INTERVAL(),
    )
    Y = np.arange(
        0,
        pipeline_romance._get_TIME_UPPER_BOUND(),
        pipeline_romance._get_TIME_INTERVAL(),
    )
    X, Y = np.meshgrid(X, Y)

    cov_arr = cov_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            X,
            Y,
            cov_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-8,
            vmax=110,
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Time (standardized)")
        plt.title("Covariance")
    else:
        surf = ax.plot_surface(
            Y,
            X,
            cov_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-8,
            vmax=110,
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Time (standardized)")
        ax.invert_xaxis()
        plt.title("Covariance")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_freq_covariance(
    cov_arr: np.array,
    rotate=False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat

    Plots a frequency covariance array.

    Args:
        cov_arr (numpy.array): The frequency covariance array to plot.
        rotate (bool, optional): Whether or not the covariance surface is rotated
                                  90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(
        0,
        pipeline_romance._get_FREQ_UPPER_BOUND(),
        pipeline_romance._get_FREQ_INTERVAL(),
    )
    Y = np.arange(
        0,
        pipeline_romance._get_FREQ_UPPER_BOUND(),
        pipeline_romance._get_FREQ_INTERVAL(),
    )
    X, Y = np.meshgrid(X, Y)

    cov_arr = cov_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            X,
            Y,
            cov_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-8,
            vmax=110,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency (Hz)")
        plt.title("Covariance")
    else:
        surf = ax.plot_surface(
            Y,
            X,
            cov_arr,
            cmap=cm.hsv_r,
            linewidth=0,
            antialiased=False,
            vmin=-8,
            vmax=110,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency (Hz)")
        ax.invert_xaxis()
        plt.title("Covariance")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_poincare_disk(
    embedding_dict: Dict,
    digit: int,
    k: int,
    language_family: enums.LanguageFamily,
    legend: bool = True,
    interp_settings: hyperbolic_analysis.InterpSettings = None,
    point_labels: bool = False,
    file_path: str = None,
):
    """Plots embeddings on a Poincare disk.

    Args:
        embedding_dict (Dict): A dictionary of embeddings as generated by hyperbolic_analysis >
                                    get_embeddings.
        digit (int): The digit depicted by the disk.
        k (int): The k value with which the disk's underlying kNN graph was constructed.
        language_family (enums.LanguageFamily): The language family for which the disk is being constructed.
        legend (bool, optional): Whether to include a legend. Defaults to True.
        interp_settings (hyperbolic_analysis.InterpSettings, optional): Settings describing interpolation
                                                                        details; will plot interpolations if
                                                                        and only if this is not None. Defaults
                                                                        to None.
        point_labels (bool, optional): Whether to label individual points. Defaults to False.
        file_path (str, optional): Where to save the image of the disk; will only save the image if not None.
                                   Defaults to None.
    """
    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    ax.set_aspect("equal", adjustable="box")

    if language_family == enums.RomanceLanguages:
        langs = utils.romance_langs
        colors = utils.colors_romance

        FR_patch = mpatches.Patch(color=colors["FR"], label="FR")
        IT_patch = mpatches.Patch(color=colors["IT"], label="IT")
        PO_patch = mpatches.Patch(color=colors["PO"], label="PO")
        SA_patch = mpatches.Patch(color=colors["SA"], label="SA")
        SI_patch = mpatches.Patch(color=colors["SI"], label="SI")

        handles = [
            FR_patch,
            IT_patch,
            PO_patch,
            SA_patch,
            SI_patch,
        ]

    elif language_family == enums.GermanicLanguages:
        langs = utils.germanic_langs
        colors = utils.colors_germanic

        # AE_patch = mpatches.Patch(color=colors["AE"], label="AE")
        BE_patch = mpatches.Patch(color=colors["BE"], label="BE")
        DT_patch = mpatches.Patch(color=colors["DT"], label="DT")
        GR_patch = mpatches.Patch(color=colors["GR"], label="GR")
        SW_patch = mpatches.Patch(color=colors["SW"], label="SW")

        handles = [
            # AE_patch,
            BE_patch,
            DT_patch,
            GR_patch,
            SW_patch,
        ]

    title_string = f"Poincare embeddings for digit {digit}; k={k}"
    plt.title(title_string)

    if interp_settings:
        fuchsia_path = mpatches.Patch(
            color=colors["cov_in"], label="Covariance interp."
        )
        handles.append(fuchsia_path)

        aquamarine_path = mpatches.Patch(
            color=colors["hyp_in"], label="Hyperbolic interp."
        )
        handles.append(aquamarine_path)

    if legend:
        plt.legend(
            handles=handles,
        )

    for lang in langs:
        x = [pt[0][0] for pt in embedding_dict[lang]]
        y = [pt[0][1] for pt in embedding_dict[lang]]

        plt.scatter(
            x,
            y,
            color=colors[lang],
            label=lang,
        )

    if interp_settings:
        anchor_1 = embedding_dict["anchor"][0][0]
        anchor_2 = embedding_dict["anchor"][1][0]

        if point_labels:
            # NOTE: Below code from https://stackoverflow.com/questions/
            #       14432557/scatter-plot-with-different-text-at-each-data-point
            # for i, txt in enumerate(txt):
            #     ax.annotate(txt, (x[i], y[i]))

            ax.annotate(
                interp_settings.lang1.value + interp_settings.speaker1.value,
                anchor_1,
            )
            ax.annotate(
                interp_settings.lang2.value + interp_settings.speaker2.value,
                anchor_2,
            )

        circle_sp1 = plt.Circle(
            (anchor_1[0], anchor_1[1]),
            0.1,
            color="black",
            fill=False,
        )
        ax.add_patch(circle_sp1)

        circle_sp2 = plt.Circle(
            (anchor_2[0], anchor_2[1]),
            0.1,
            color="black",
            fill=False,
        )
        ax.add_patch(circle_sp2)

        cov_interps = embedding_dict["in"]
        hyp_interps = hyperbolic_analysis.poincare_linspace(
            u=anchor_1,
            v=anchor_2,
        )[
            1:-1
        ]  # Trim anchor points off of interpolations

        for intp in cov_interps:
            plt.scatter(
                [intp[0][0]],
                [intp[0][1]],
                color=colors["cov_in"],
                label="Cov interp.",
            )

        for intp in hyp_interps:
            plt.scatter(
                [intp[0]],
                [intp[1]],
                color=colors["hyp_in"],
                label="Hyp interp.",
            )

    if not file_path:
        plt.show()

    else:
        if interp_settings:
            plt.savefig(
                f"{file_path}/"
                + f"{interp_settings.lang1.value}{interp_settings.speaker1.value}->"
                + f"{interp_settings.lang2.value}{interp_settings.speaker2.value}_{digit}"
            )
        else:
            plt.savefig(f"{file_path}/{digit}")

    plt.clf()

    if interp_settings:
        interp_score = utils._interpolation_score(
            anchor_1=anchor_1,
            anchor_2=anchor_2,
            cov_interps=cov_interps,
            hyp_interps=hyp_interps,
        )

        coord_high = []
        coord_low = []

        for lang in langs:
            embedding_list = embedding_dict[lang]
            for cl, ch, _ in embedding_list:
                coord_high.append(ch)
                coord_low.append(cl)

        # Handle interpolations too
        embedding_list = embedding_dict["in"]
        for cl, ch, txt in embedding_list:
            # Below statement excludes anchor points
            if txt == "":
                coord_high.append(ch)
                coord_low.append(cl)

        Qlocal, Qglobal, _ = embedding_scoring.get_quality_metrics(
            coord_high=coord_high,
            coord_low=coord_low,
            k_neighbours=k,
        )

        Qlocal = round(Qlocal, 4)
        Qglobal = round(Qglobal, 4)

        return (interp_score, Qlocal, Qglobal)


def plot_poincare_centroids(
    centroid_dict: Dict[str, np.array],
    digit: int,
    k: int,
    language_family: enums.LanguageFamily,
    kernel_sigma: float = 100,
    legend: bool = True,
) -> None:
    """Plots the per-language centroids in a Poincare disk.

    Args:
        centroid_dict (Dict[str, np.array]): A dictionary of languages and their
                                             centroids.
        digit (int): The digit depicted by the disk.
        k (int): The k value with which the disk's underlying kNN graph was constructed.
        language_family (enums.LanguageFamily): The language family for which centroids are
                                                being plotted.
        kernel_sigma (float, optional): The kernel_sigma value with which the disk's underlying kNN
                                        graph was constructed. Defaults to 100.
        legend (bool, optional): Whether to include a legend. Defaults to True.
    """
    plt.title(
        f"Poincare embeddings for digit {digit}; k={k}, kernel_sigma={kernel_sigma}"
    )

    if language_family == enums.RomanceLanguages:
        colors = utils.colors_romance

        FR_patch = mpatches.Patch(color=colors["FR"], label="FR")
        IT_patch = mpatches.Patch(color=colors["IT"], label="IT")
        PO_patch = mpatches.Patch(color=colors["PO"], label="PO")
        SA_patch = mpatches.Patch(color=colors["SA"], label="SA")
        SI_patch = mpatches.Patch(color=colors["SI"], label="SI")

        if legend:
            handles = [
                FR_patch,
                IT_patch,
                PO_patch,
                SA_patch,
                SI_patch,
            ]

            plt.legend(handles=handles)

    elif language_family == enums.GermanicLanguages:
        colors = utils.colors_germanic

        # AE_patch = mpatches.Patch(color=colors["AE"], label="AE")
        BE_patch = mpatches.Patch(color=colors["BE"], label="BE")
        DT_patch = mpatches.Patch(color=colors["DT"], label="DT")
        GR_patch = mpatches.Patch(color=colors["GR"], label="GR")
        SW_patch = mpatches.Patch(color=colors["SW"], label="SW")

        if legend:
            handles = [
                # AE_patch,
                BE_patch,
                DT_patch,
                GR_patch,
                SW_patch,
            ]

            plt.legend(handles=handles)

    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    ax.set_aspect("equal", adjustable="box")

    if language_family == enums.RomanceLanguages:
        plt.scatter(
            centroid_dict["FR"][0],
            centroid_dict["FR"][1],
            color=colors["FR"],
            label="FR",
        )
        plt.scatter(
            centroid_dict["IT"][0],
            centroid_dict["IT"][1],
            color=colors["IT"],
            label="IT",
        )
        plt.scatter(
            centroid_dict["PO"][0],
            centroid_dict["PO"][1],
            color=colors["PO"],
            label="PO",
        )
        plt.scatter(
            centroid_dict["SA"][0],
            centroid_dict["SA"][1],
            color=colors["SA"],
            label="SA",
        )
        plt.scatter(
            centroid_dict["SI"][0],
            centroid_dict["SI"][1],
            color=colors["SI"],
            label="SI",
        )

    elif language_family == enums.GermanicLanguages:
        # plt.scatter(
        #     centroid_dict["AE"][0],
        #     centroid_dict["AE"][1],
        #     color=colors["AE"],
        #     label="AE",
        # )
        plt.scatter(
            centroid_dict["BE"][0],
            centroid_dict["BE"][1],
            color=colors["BE"],
            label="BE",
        )
        plt.scatter(
            centroid_dict["DT"][0],
            centroid_dict["DT"][1],
            color=colors["DT"],
            label="DT",
        )
        plt.scatter(
            centroid_dict["GR"][0],
            centroid_dict["GR"][1],
            color=colors["GR"],
            label="GR",
        )
        plt.scatter(
            centroid_dict["SW"][0],
            centroid_dict["SW"][1],
            color=colors["SW"],
            label="SW",
        )

    plt.show()


def plot_aligned_digit_centroids(
    embedding_dict: Dict[str, List],
    k: int,
    language_family: enums.LanguageFamily,
    legend: bool = True,
    radii: bool = True,
) -> None:
    """Aligns and plots the centroid for each language/digit pair in the same Poincare
    disk.

    Args:
        embedding_dict (Dict[str, List]): Embeddings returned by hyperbolic_analysis.py >
                                          align_all_digit_disks.

        k (int): The k value with which the disk's underlying kNN graph was constructed.
        language_family (enums.LanguageFamily): The language family for which aligned disks
                                                are being plotted
        legend (bool, optional): Whether to include a legend. Defaults to True.
        radii (bool, optional): Whether to include visual indicators of each language's
                                overall centroid. Defaults to True.

    Returns:
        None.
    """
    plt.title(f"Aligned embeddings; k={k}")

    alpha = 0.5

    if language_family == enums.RomanceLanguages:
        langs = utils.romance_langs

        colors = utils.colors_romance

        FR_patch = mpatches.Patch(color=colors["FR"], label="FR")
        IT_patch = mpatches.Patch(color=colors["IT"], label="IT")
        PO_patch = mpatches.Patch(color=colors["PO"], label="PO")
        SA_patch = mpatches.Patch(color=colors["SA"], label="SA")
        SI_patch = mpatches.Patch(color=colors["SI"], label="SI")

        if legend:
            handles = [
                FR_patch,
                IT_patch,
                PO_patch,
                SA_patch,
                SI_patch,
            ]

            plt.legend(handles=handles)

    elif language_family == enums.GermanicLanguages:
        langs = utils.germanic_langs

        colors = utils.colors_germanic

        # AE_patch = mpatches.Patch(color=colors["AE"], label="AE")
        BE_patch = mpatches.Patch(color=colors["BE"], label="BE")
        DT_patch = mpatches.Patch(color=colors["DT"], label="DT")
        GR_patch = mpatches.Patch(color=colors["GR"], label="GR")
        SW_patch = mpatches.Patch(color=colors["SW"], label="SW")

        if legend:
            handles = [
                # AE_patch,
                BE_patch,
                DT_patch,
                GR_patch,
                SW_patch,
            ]

            plt.legend(handles=handles)

    for lang in langs:
        plt.scatter(
            [x[0][0] for x in embedding_dict[lang]],
            [x[0][1] for x in embedding_dict[lang]],
            color=colors[lang],
            label=lang,
        )

    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    centroids = hyperbolic_analysis.get_poincare_centroids(
        embedding_dict=embedding_dict,
        language_family=language_family,
    )

    if radii:
        # NOTE: radii dict defined as such to allow for e.g. defining radii
        #       to correspond to point cluster diameter, etc.

        if language_family == enums.RomanceLanguages:
            circle1 = plt.Circle(
                centroids["FR"],
                0.07,
                alpha=alpha,
                color=colors["FR"],
                fill=True,
            )
            ax.add_patch(circle1)

            circle2 = plt.Circle(
                centroids["IT"],
                0.07,
                alpha=alpha,
                color=colors["IT"],
                fill=True,
            )
            ax.add_patch(circle2)

            circle3 = plt.Circle(
                centroids["PO"],
                0.07,
                alpha=alpha,
                color=colors["PO"],
                fill=True,
            )
            ax.add_patch(circle3)

            circle4 = plt.Circle(
                centroids["SA"],
                0.07,
                alpha=alpha,
                color=colors["SA"],
                fill=True,
            )
            ax.add_patch(circle4)

            circle5 = plt.Circle(
                centroids["SI"],
                0.07,
                alpha=alpha,
                color=colors["SI"],
                fill=True,
            )
            ax.add_patch(circle5)

        elif language_family == enums.GermanicLanguages:
            # circle1 = plt.Circle(
            #     centroids["AE"],
            #     0.07,
            #     alpha=alpha,
            #     color=colors["AE"],
            #     fill=True,
            # )
            # ax.add_patch(circle1)

            circle2 = plt.Circle(
                centroids["BE"],
                0.07,
                alpha=alpha,
                color=colors["BE"],
                fill=True,
            )
            ax.add_patch(circle2)

            circle3 = plt.Circle(
                centroids["DT"],
                0.07,
                alpha=alpha,
                color=colors["DT"],
                fill=True,
            )
            ax.add_patch(circle3)

            circle4 = plt.Circle(
                centroids["GR"],
                0.07,
                alpha=alpha,
                color=colors["GR"],
                fill=True,
            )
            ax.add_patch(circle4)

            circle5 = plt.Circle(
                centroids["SW"],
                0.07,
                alpha=alpha,
                color=colors["SW"],
                fill=True,
            )
            ax.add_patch(circle5)

    ax.set_aspect("equal", adjustable="box")
    plt.show()


def minmax_scale_dict(
    d: dict,
):
    """Applies minmax scaling to a dictionary of distances.

    Args:
        d (dict): A dictionary of dictionaries that takes the following form:

        {
            "lang1": {
                "lang2": <float dist>,
                "lang3": <float dist>,
            },
            "lang2": {
                "lang1": <float dist>,
                "lang3": <float dist>,
            }
            ...
        }

    Returns:
        dict: The dictionary with minmax scaling applied.
    """
    vals = np.array(
        [[x for x in subdict.values() if x is not None] for subdict in d.values()]
    ).reshape(
        -1,
    )

    lang_min = np.min(vals)
    lang_max = np.max(vals)

    ret_d = deepcopy(d)

    for subdict in ret_d.values():
        for lang in subdict:
            if subdict[lang] is not None:
                subdict[lang] = (subdict[lang] - lang_min) / (lang_max - lang_min)
            else:
                subdict[lang] = np.nan

    return ret_d


def get_mse(
    covar_dict: dict,
    poincare_dict: dict,
):
    """Gets the error between two distance dictionaries.

    Args:
        covar_dict (dict): A dictionary of Procrustes distances.
        poincare_dict (dict): A dictionary of Poincaré distances.

        The dictionaries should have the following form:

        {
            "lang1": {
                "lang2": <float dist>,
                "lang3": <float dist>,
            },
            "lang2": {
                "lang1": <float dist>,
                "lang3": <float dist>,
            }
            ...
        }

    Returns:
        float: The error between the sets of distances.
    """
    covar_langs = list(covar_dict.keys())
    poincare_langs = list(poincare_dict.keys())

    if covar_langs != poincare_langs:
        raise Exception("Make sure dicts contain the same languages")

    count = 0
    total_squared_err = 0

    for i in range(len(covar_langs)):
        for j in range(i, len(poincare_langs)):
            lang_a = covar_langs[i]
            lang_b = poincare_langs[j]

            if lang_a != lang_b:
                total_squared_err += (
                    covar_dict[lang_a][lang_b] - poincare_dict[lang_a][lang_b]
                ) ** 2
                count += 1

    return total_squared_err / count


def compare_lang_space_divergence(
    cov_dist_dict: dict,
    poincare_dist_dict,
    hyperparam_range: range,
    title: str,
):
    """Compares the divergence between a Procrustes and Poincaré language space.

    Args:
        cov_dist_dict (dict): A dictionary of Procrustes distances.
        poincare_dist_dict (dict):  A dictionary of Poincaré distances.
        hyperparam_range (range): A range of k values for which the divergences will be tested.
        title (str): The title of the resulting plot.

        The dictionaries should have the following form:

        {
            "lang1": {
                "lang2": <float dist>,
                "lang3": <float dist>,
            },
            "lang2": {
                "lang1": <float dist>,
                "lang3": <float dist>,
            }
            ...
        }
    """
    mse_arr = []

    for k in hyperparam_range:
        mse_arr.append(
            get_mse(
                covar_dict=minmax_scale_dict(cov_dist_dict),
                poincare_dict=minmax_scale_dict(poincare_dist_dict.get(k)),
            )
        )

    plt.scatter([x for x in range(len(mse_arr))], mse_arr)
    plt.plot(
        [np.mean(mse_arr) for _ in range(len(mse_arr))], ":", color="r", label="Average"
    )
    plt.title(
        title,
    )
    plt.xlabel("k")
    plt.xticks(
        ticks=[x for x in range(len(mse_arr))], labels=[x for x in hyperparam_range]
    )
    plt.ylabel("Divergence")
    plt.legend()
