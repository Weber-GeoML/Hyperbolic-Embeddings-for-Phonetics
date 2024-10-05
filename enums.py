from aenum import Enum


class LanguageFamily(Enum):
    """Parent class for language enums"""


class RomanceLanguages(LanguageFamily):
    """Holds constant string representations for each Romance Language"""

    FRENCH = "FR"
    ITALIAN = "IT"
    _BRAZILIAN_PORTUGUESE = "PB"
    _LUSITANIAN_PORTUGUESE = "PT"
    PORTUGUESE = "PO"
    AMERICAN_SPANISH = "SA"
    IBERIAN_SPANISH = "SI"


class GermanicLanguages(LanguageFamily):
    """Holds constant string representations for each Germanic Language"""

    AMERICAN_ENGLISH = "AE"
    BRITISH_AUSTRALIAN_ENGLISH = "BE"
    DUTCH = "DT"
    GERMAN = "GR"
    SWEDISH = "SW"


class Speakers(Enum):
    """Parent class for ____Speakers Enums"""


class CovarianceType(Enum):
    """Holds constant string representations for each covariance type"""

    TIME = "time"
    FREQUENCY = "freq"


class FrenchSpeakers(Speakers):
    """Holds constant string representations for the French speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_5 = "05"
    SPEAKER_6 = "06"
    SPEAKER_7 = "07"
    SPEAKER_8 = "08"


class ItalianSpeakers(Speakers):
    """Holds constant string representations for the Italian speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"


class _BrazilianPortugueseSpeakers(Speakers):
    """Holds constant string representations for the Brazilian Portuguese speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"
    SPEAKER_6 = "06"


class _LusitanianPortugueseSpeakers(Speakers):
    """Holds constant string representations for the Lusitanian Portuguese speakers in the
    dataset
    """

    SPEAKER_2 = "02"
    SPEAKER_3 = "03"


class PortugueseSpeakers(Speakers):
    """Holds constant string representations for the Portuguese speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"
    SPEAKER_6 = "06"


class AmericanSpanishSpeakers(Speakers):
    """Holds constant string representations for the American Spanish speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"


class IberianSpanishSpeakers(Speakers):
    """Holds constant string representations for the Iberian Spanish speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"
    SPEAKER_6 = "06"
    SPEAKER_7 = "07"


class AmericanEnglishSpeakers(Speakers):
    """Holds constant string representations for the American English speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"


class BritishAustralianEnglishSpeakers(Speakers):
    """Holds constant string representations for the British/Australian English speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"


class DutchSpeakers(Speakers):
    """Holds constant string representations for the Dutch speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"


class GermanSpeakers(Speakers):
    """Holds constant string representations for the German speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"


class SwedishSpeakers(Speakers):
    """Holds constant string representations for the Swedish speakers in the
    dataset
    """

    SPEAKER_1 = "01"
    SPEAKER_2 = "02"
    SPEAKER_3 = "03"
    SPEAKER_4 = "04"
    SPEAKER_5 = "05"
    SPEAKER_6 = "06"


class _FrenchDict(Enum):
    FRENCH_1 = "un"
    FRENCH_2 = "deux"
    FRENCH_3 = "trois"
    FRENCH_4 = "quatre"
    FRENCH_5 = "cinq"
    FRENCH_6 = "six"
    FRENCH_7 = "sept"
    FRENCH_8 = "huit"
    FRENCH_9 = "neuf"
    FRENCH_10 = "dix"


class _ItalianDict(Enum):
    ITALIAN_1 = "uno"
    ITALIAN_2 = "due"
    ITALIAN_3 = "tre"
    ITALIAN_4 = "quattro"
    ITALIAN_5 = "cinque"
    ITALIAN_6 = "sei"
    ITALIAN_7 = "sette"
    ITALIAN_8 = "otto"
    ITALIAN_9 = "nove"
    ITALIAN_10 = "dieci"


class _BrazilianPortugueseDict(Enum):
    BRAZILIAN_PORTUGUESE_1 = "um"
    BRAZILIAN_PORTUGUESE_2 = "dois"
    BRAZILIAN_PORTUGUESE_3 = "tres"
    BRAZILIAN_PORTUGUESE_4 = "quatro"
    BRAZILIAN_PORTUGUESE_5 = "cinco"
    BRAZILIAN_PORTUGUESE_6 = "seis"
    BRAZILIAN_PORTUGUESE_7 = "sete"
    BRAZILIAN_PORTUGUESE_8 = "oito"
    BRAZILIAN_PORTUGUESE_9 = "nove"
    BRAZILIAN_PORTUGUESE_10 = "dez"


class _LusitanianPortugueseDict(Enum):
    LUSITANIAN_PORTUGUESE_1 = "um"
    LUSITANIAN_PORTUGUESE_2 = "dois"
    LUSITANIAN_PORTUGUESE_3 = "tres"
    LUSITANIAN_PORTUGUESE_4 = "quatro"
    LUSITANIAN_PORTUGUESE_5 = "cinco"
    LUSITANIAN_PORTUGUESE_6 = "seis"
    LUSITANIAN_PORTUGUESE_7 = "sete"
    LUSITANIAN_PORTUGUESE_8 = "oito"
    LUSITANIAN_PORTUGUESE_9 = "nove"
    LUSITANIAN_PORTUGUESE_10 = "dez"


class _AmericanSpanishDict(Enum):
    AMERICAN_SPANISH_1 = "uno"
    AMERICAN_SPANISH_2 = "dos"
    AMERICAN_SPANISH_3 = "tres"
    AMERICAN_SPANISH_4 = "cuatro"
    AMERICAN_SPANISH_5 = "cinco"
    AMERICAN_SPANISH_6 = "seis"
    AMERICAN_SPANISH_7 = "siete"
    AMERICAN_SPANISH_8 = "ocho"
    AMERICAN_SPANISH_9 = "nueve"
    AMERICAN_SPANISH_10 = "diez"


class _IberianSpanishDict(Enum):
    IBERIAN_SPANISH_1 = "uno"
    IBERIAN_SPANISH_2 = "dos"
    IBERIAN_SPANISH_3 = "tres"
    IBERIAN_SPANISH_4 = "cuatro"
    IBERIAN_SPANISH_5 = "cinco"
    IBERIAN_SPANISH_6 = "seis"
    IBERIAN_SPANISH_7 = "siete"
    IBERIAN_SPANISH_8 = "ocho"
    IBERIAN_SPANISH_9 = "nueve"
    IBERIAN_SPANISH_10 = "diez"


class _AmericanEnglishDict(Enum):
    AMERICAN_ENGLISH_1 = "one"
    AMERICAN_ENGLISH_2 = "two"
    AMERICAN_ENGLISH_3 = "three"
    AMERICAN_ENGLISH_4 = "four"
    AMERICAN_ENGLISH_5 = "five"
    AMERICAN_ENGLISH_6 = "six"
    AMERICAN_ENGLISH_7 = "seven"
    AMERICAN_ENGLISH_8 = "eight"
    AMERICAN_ENGLISH_9 = "nine"
    AMERICAN_ENGLISH_10 = "ten"


class _BritishAustralianEnglishDict(Enum):
    BRITISH_AUSTRALIAN_ENGLISH_1 = "one"
    BRITISH_AUSTRALIAN_ENGLISH_2 = "two"
    BRITISH_AUSTRALIAN_ENGLISH_3 = "three"
    BRITISH_AUSTRALIAN_ENGLISH_4 = "four"
    BRITISH_AUSTRALIAN_ENGLISH_5 = "five"
    BRITISH_AUSTRALIAN_ENGLISH_6 = "six"
    BRITISH_AUSTRALIAN_ENGLISH_7 = "seven"
    BRITISH_AUSTRALIAN_ENGLISH_8 = "eight"
    BRITISH_AUSTRALIAN_ENGLISH_9 = "nine"
    BRITISH_AUSTRALIAN_ENGLISH_10 = "ten"


class _DutchDict(Enum):
    DUTCH_1 = "een"
    DUTCH_2 = "twee"
    DUTCH_3 = "drie"
    DUTCH_4 = "vier"
    DUTCH_5 = "vijf"
    DUTCH_6 = "zes"
    DUTCH_7 = "zeven"
    DUTCH_8 = "acht"
    DUTCH_9 = "negen"
    DUTCH_10 = "tien"


class _GermanDict(Enum):
    GERMAN_1 = "eins"
    GERMAN_2 = "zwei"
    GERMAN_3 = "drei"
    GERMAN_4 = "vier"
    GERMAN_5 = "fünf"
    GERMAN_6 = "sechs"
    GERMAN_7 = "sieben"
    GERMAN_8 = "acht"
    GERMAN_9 = "neun"
    GERMAN_10 = "zehn"


class _SwedishDict(Enum):
    SWEDISH_1 = "ett"
    SWEDISH_2 = "två"
    SWEDISH_3 = "tre"
    SWEDISH_4 = "fyra"
    SWEDISH_5 = "fem"
    SWEDISH_6 = "sex"
    SWEDISH_7 = "sju"
    SWEDISH_8 = "åtta"
    SWEDISH_9 = "nio"
    SWEDISH_10 = "tio"


class _DBType(Enum):
    GLOBAL_INVERSE_WARPING_FUNCTION = "global_inverse_warping_function"
    RAW_SPECTROGRAM = "raw_spectrogram"
    TIME_ALIGNED_SPECTROGRAM = "time_aligned_spectrogram"
    TIME_ALIGNED_SMOOTHED_SPECTROGRAM = "time_aligned_smoothed_spectrogram"
    MEAN_SPECTROGRAM = "mean_spectrogram"
    RESIDUAL_SPECTROGRAM = "residual_spectrogram"
