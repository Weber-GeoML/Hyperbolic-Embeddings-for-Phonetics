import enums
import utils

from sqlite3 import connect, OperationalError
import autograd.numpy as np
from typing import Dict, List, Tuple, Callable


def get_raw_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the raw spectrogram array corresponding to the given speaker and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer in the range [1, 10] (inclusive).

    Returns:
        np.array: A NumPy array containing the desired raw spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.RAW_SPECTROGRAM.value}.db",
        language_family,
        language,
        speaker,
        digit,
    )


def get_time_aligned_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the time aligned spectrogram array corresponding to the given speaker and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer in the range [1, 10] (inclusive).

    Returns:
        np.array: A NumPy array containing the desired time aligned spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.TIME_ALIGNED_SPECTROGRAM.value}.db",
        language_family,
        language,
        speaker,
        digit,
    )


def get_time_aligned_smoothed_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the time aligned and smoothed spectrogram array corresponding to the given speaker and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer 1 through 10 (inclusive).

    Returns:
        np.array: A NumPy array containing the desired time aligned and smoothed spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value}.db",
        language_family,
        language,
        speaker,
        digit,
    )


def get_mean_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
    digit: int,
):
    """Gets the mean spectrogram array corresponding to the given language and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.
        digit (int): The digit corresponding to the desired array.

    Returns:
        np.array: A NumPy array containing the desired mean spectrogram array.
    """
    if language_family == enums.RomanceLanguages:
        family = "romance"
    elif language_family == enums.GermanicLanguages:
        family = "germanic"
    else:
        raise Exception(
            f"language family: {language_family} is invalid. Must be either enums.RomanceLanguages or enums.GermanicLanguages"
        )

    language = language.value

    with connect(f"dbs/{family}/{enums._DBType.MEAN_SPECTROGRAM.value}.db") as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}_{digit}").fetchall()
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(
                    f'table: "{language}_{digit}" does not exist in this table.'
                )
        arr = np.array(cur.execute(f"SELECT * FROM {language}_{digit}").fetchall())

    return arr


def get_residual_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the residual spectrogram array corresponding to the given speaker and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer 1 through 10 (inclusive).

    Returns:
        np.array: A NumPy array containing the desired residual spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db",
        language_family,
        language,
        speaker,
        digit,
    )


def get_mean_residual_spectrogram_array(
    language_family: enums.LanguageFamily,
    language: enums.LanguageFamily,
) -> np.array:
    """Gets the mean residual spectrogram array corresponding to the given speaker and digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        language (enums.LanguageFamily): The language corresponding to the desired spectrogram array.

    Returns:
        np.array: A NumPy array containing the desired mean residual spectrogram array.
    """

    if language_family == enums.RomanceLanguages:
        family = "romance"
    elif language_family == enums.GermanicLanguages:
        family = "germanic"
    else:
        raise Exception(
            f"language family: {language_family} is invalid. Must be either enums.RomanceLanguages or enums.GermanicLanguages"
        )

    language = language.value

    with connect(f"dbs/{family}/mean_residual_spectrogram.db") as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}").fetchall()
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(f'language: "{language}" does not exist in this table.')
        arr = np.array(cur.execute(f"SELECT * FROM {language}").fetchall())

    return arr


def get_digit_spectrogram_arrays(
    language_family: enums.LanguageFamily,
    digit: int,
    db_type: enums._DBType = enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value,
) -> List[Tuple[str, np.array]]:
    """Gets the spectrograms for all pronunciations of a given digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        digit (int): The digit in question.
        db_type (enums._DBType, optional): The database from which to select the spectrograms. Defaults to
                                           enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value.

    Returns:
        List[Tuple[str, np.array]]: A list of tuples, where each tuple consists of a label and a
                                    spectrogram.
    """
    if language_family == enums.RomanceLanguages:
        family = "romance"
    elif language_family == enums.GermanicLanguages:
        family = "germanic"
    else:
        raise Exception(
            f"language family: {language_family} is invalid. Must be either enums.RomanceLanguages or enums.GermanicLanguages"
        )

    with connect(f"dbs/{family}/{db_type}.db") as c:
        cur = c.cursor()
        res = cur.execute(
            f"SELECT name FROM sqlite_schema WHERE type='table' \
              AND name NOT LIKE 'sqlite_%';"
        ).fetchall()

        arrs = [
            (
                x[0],
                utils._get_array(
                    db_str=f"{db_type}.db",
                    language_family=language_family,
                    language=x[0][:2],
                    speaker=x[0][2:4],
                    digit=digit,
                    override=True,
                ),
            )
            for x in res
            # Generalizing the slice for digit 10
            if x[0][-((digit // 10) + 1) :] == f"{digit}"
        ]

        # NOTE: Below is a fix to get around some apparently missing data
        return [x for x in arrs if x[1] != np.array([])]


def get_digit_mean_spectrogram_array(
    language_family: enums.LanguageFamily,
    digit: int,
) -> np.array:
    """Gets the mean spectrogram across all pronunciations, regardless of language, for a given digit.

    Args:
        language_family (enums.LanguageFamily): The language family of the desired spectrogram (i.e., 
                                                either Romance or Germanic).
        digit (int): The digit corresponding to the mean being calculated.

    Returns:
        np.array: The mean spectrogram across all pronunciations, regardless of language, for a given digit.
    """
    arrs = [arr for _, arr in get_digit_spectrogram_arrays(language_family=language_family, digit=digit)]
    return sum(arrs) / len(arrs)
