import pipeline_utils

import numpy as np
from sqlite3 import connect
import enums
from math import sqrt


def _get_TIME_INTERVAL() -> float:
    """Getter method for TIME_INTERVAL.

    Returns:
        float: TIME_INTERVAL.
    """
    return pipeline_utils.TIME_INTERVAL


def _get_TIME_UPPER_BOUND() -> float:
    """Getter method for TIME_UPPER_BOUND.

    Returns:
        float: TIME_UPPER_BOUND.
    """
    return pipeline_utils.TIME_UPPER_BOUND


def _get_FREQ_INTERVAL() -> float:
    """Getter method for FREQ_INTERVAL.

    Returns:
        float: FREQ_INTERVAL.
    """
    return pipeline_utils.FREQ_INTERVAL


def _get_FREQ_UPPER_BOUND() -> float:
    """Getter method for FREQ_UPPER_BOUND.

    Returns:
        float: FREQ_UPPER_BOUND.
    """
    return pipeline_utils.FREQ_UPPER_BOUND


def _create_mean_spectrograms():
    """Creates mean spectrograms and stores the results.

    Args:
        None.

    Returns:
        None.
    """
    print("NOW CREATING MEAN SPECTROGRAMS")
    with connect(
        f"{enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value}.db"
    ) as tas_c:
        tas_cur = tas_c.cursor()
        for dgt in range(1, 11):
            for lang in [
                l.value
                for l in enums.RomanceLanguages
                if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
                and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
            ]:
                mean_spct = np.zeros(
                    (
                        int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL),
                        int(pipeline_utils.FREQ_UPPER_BOUND / pipeline_utils.FREQ_INTERVAL),
                    )
                )
                counter = 0
                if lang == enums.RomanceLanguages.FRENCH.value:
                    for spkr in [s.value for s in enums.FrenchSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.ITALIAN.value:
                    for spkr in [s.value for s in enums.ItalianSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.PORTUGUESE.value:
                    for spkr in [
                        s.value for s in enums._BrazilianPortugueseSpeakers
                    ] + [s.value for s in enums._LusitanianPortugueseSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
                    for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
                    for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1

                mean_spct /= counter

                with connect(f"{enums._DBType.MEAN_SPECTROGRAM.value}.db") as mean_c:
                    mean_cur = mean_c.cursor()
                    for t in range(0, int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL)):
                        hz_values = mean_spct[t]
                        insert_query_string = f"INSERT INTO {lang}_{dgt} VALUES ("
                        for hz in hz_values:
                            insert_query_string += f"{hz}, "
                        insert_query_string = insert_query_string[:-2] + ");"
                        mean_cur.execute(insert_query_string)
                    mean_c.commit()
    print("DONE CREATING MEAN SPECTROGRAMS")


def _create_mean_residual_spectrograms():
    """Creates mean residual spectrograms and stores the results.

    Args:
        None.

    Returns:
        None.
    """
    print("NOW CREATING MEAN RESIDUAL SPECTROGRAMS")
    with connect(f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db") as rs_c:
        rs_cur = rs_c.cursor()
        for lang in [
            l.value
            for l in enums.RomanceLanguages
            if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
            and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
        ]:
            res_mean_spct = np.zeros(
                (
                    int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL),
                    int(pipeline_utils.FREQ_UPPER_BOUND / pipeline_utils.FREQ_INTERVAL),
                )
            )
            counter = 0
            if lang == enums.RomanceLanguages.FRENCH.value:
                for spkr in [s.value for s in enums.FrenchSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.ITALIAN.value:
                for spkr in [s.value for s in enums.ItalianSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.PORTUGUESE.value:
                for spkr in [s.value for s in enums._BrazilianPortugueseSpeakers] + [
                    s.value for s in enums._LusitanianPortugueseSpeakers
                ]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
                for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
                for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1

            # NOTE: taking the sample mean here
            res_mean_spct /= counter - 1

            with connect("mean_residual_spectrogram.db") as mean_c:
                mean_cur = mean_c.cursor()
                for t in range(0, int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL)):
                    hz_values = res_mean_spct[t]
                    insert_query_string = f"INSERT INTO {lang} VALUES ("
                    for hz in hz_values:
                        insert_query_string += f"{hz}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    mean_cur.execute(insert_query_string)
                mean_c.commit()
    print("DONE CREATING MEAN RESIDUAL SPECTROGRAMS")


def _calculate_covariance(
    language: enums.RomanceLanguages,
    cov_type="",
) -> None:
    """Calculates a covariance structure.

    Args:
        language (enums.RomanceLanguages): The language for which to calculate
                                           the covariance structure.
        cov_type (str, optional): The type of covariance to calculate. Defaults
                                  to "".

    Returns:
        None.
    """
    # type checks
    if type(language) != enums.RomanceLanguages:
        raise Exception(f"language: {language} is not a valid RomanceLanguages enum")

    if language == enums.RomanceLanguages.ITALIAN:
        speakers = enums.ItalianSpeakers
    elif language == enums.RomanceLanguages.FRENCH:
        speakers = enums.FrenchSpeakers
    elif language == enums.RomanceLanguages.PORTUGUESE:
        speakers = enums.PortugueseSpeakers
    elif language == enums.RomanceLanguages.AMERICAN_SPANISH:
        speakers = enums.AmericanSpanishSpeakers
    elif language == enums.RomanceLanguages.IBERIAN_SPANISH:
        speakers = enums.IberianSpanishSpeakers

    language = language.value

    if cov_type == "time":
        covariance = np.zeros(
            (
                int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL),
                int(pipeline_utils.TIME_UPPER_BOUND / pipeline_utils.TIME_INTERVAL),
            )
        )
        UPPER_BOUND = pipeline_utils.TIME_UPPER_BOUND
        INTERVAL = pipeline_utils.TIME_INTERVAL
    elif cov_type == "freq":
        covariance = np.zeros(
            (
                int(pipeline_utils.FREQ_UPPER_BOUND / pipeline_utils.FREQ_INTERVAL),
                int(pipeline_utils.FREQ_UPPER_BOUND / pipeline_utils.FREQ_INTERVAL),
            )
        )
        UPPER_BOUND = pipeline_utils.FREQ_UPPER_BOUND
        INTERVAL = pipeline_utils.FREQ_INTERVAL
    else:
        raise Exception(
            f"cov_type: {cov_type} invalid. Choose either 'time' or 'freq'."
        )

    conn = connect("mean_residual_spectrogram.db")
    cur = conn.cursor()

    mean_arr = np.array(cur.execute(f"SELECT * FROM {language}").fetchall())

    conn = connect(f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db")
    cur = conn.cursor()

    # Taking covariance over all speaker/word combos in a given language
    language_arrays = []

    for dgt in range(1, 11):
        for spkr in speakers:
            arr = np.array(
                cur.execute(
                    f"SELECT * FROM {language}{spkr.value}_word{dgt}"
                ).fetchall()
            )
            if arr.shape != (0,):
                language_arrays.append(arr)

    n_L = len(language_arrays)

    if cov_type == "time":
        for time_1 in np.arange(0, UPPER_BOUND, INTERVAL):
            for time_2 in np.arange(0, UPPER_BOUND, INTERVAL):
                cov = (1 / (n_L - 1)) * sum(
                    [
                        pipeline_utils._integrate_residual_mean_difference_time(
                            int(time_1 / INTERVAL),
                            int(time_2 / INTERVAL),
                            arr,
                            mean_arr,
                        )
                        for arr in language_arrays
                    ]
                )

                covariance[int(time_1 / INTERVAL)][int(time_2 / INTERVAL)] = cov

    elif cov_type == "freq":
        for omega_1 in np.arange(0, UPPER_BOUND, INTERVAL):
            for omega_2 in np.arange(0, UPPER_BOUND, INTERVAL):
                cov = (1 / (n_L - 1)) * sum(
                    [
                        pipeline_utils._integrate_residual_mean_difference_freq(
                            int(omega_1 / INTERVAL),
                            int(omega_2 / INTERVAL),
                            arr,
                            mean_arr,
                        )
                        for arr in language_arrays
                    ]
                )

                covariance[int(omega_1 / INTERVAL)][int(omega_2 / INTERVAL)] = cov

    covariance /= sqrt(np.trace(covariance))

    conn = connect(f"{cov_type}_covariance.db")
    cur = conn.cursor()

    for t in range(0, int(UPPER_BOUND / INTERVAL)):
        vals = covariance[t]
        insert_query_string = f"INSERT INTO {language} VALUES ("
        for val in vals:
            insert_query_string += f"{val}, "
        insert_query_string = insert_query_string[:-2] + ");"
        cur.execute(insert_query_string)
    conn.commit()


def _calculate_covariances() -> None:
    """Calculates the time and frequency covariance for all languages.

    Returns:
        None.
    """
    for lang in [
        l
        for l in enums.RomanceLanguages
        if l != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE
        and l != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE
    ]:
        print(f"NOW CALCULATING TIME COVARIANCE FOR LANGUAGE: {lang.value}")
        _calculate_covariance(language=lang, cov_type="time")
        print(f"DONE CALCULATING TIME COVARIANCE FOR LANGUAGE: {lang.value}")

        print("")

        print(f"NOW CALCULATING FREQ COVARIANCE FOR LANGUAGE: {lang.value}")
        _calculate_covariance(language=lang, cov_type="freq")
        print(f"DONE CALCULATING FREQ COVARIANCE FOR LANGUAGE: {lang.value}")

        print("")
