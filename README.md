# Analyzing Language Families via Hyperbolic Embeddings of Phonetic Data

#### Patrick McDonald and Melanie Weber

<br>

## Requirements

Before running code from this repo, create the [conda][1] environment specified in `hyp_phon_env.yml`:

```
conda env create -f hyp_phon_env.yml
```

and then activate your new environment:

```
conda activate hyp_phon_env
```

Check that the above worked by running the following:

```
conda env list
```

You should see `hyp_phon_env` listed among your environments and marked with a `*` as your active environment.

<br>

## Experiments and Visualization

### Interacting With Spectrograms

[Spectrograms][2] are the basic object of this paper's analysis. Here, each spectrogram surface characterizes a speaker's pronunciation of a digit one through ten in their language. For instance, you can get a NumPy array containing the spectrogram for American Spanish speaker \#3 saying the word for "3":

```
spgm = spectrograms.get_time_aligned_smoothed_spectrogram_array(
    language_family=enums.RomanceLanguages,
    language=enums.RomanceLanguages.AMERICAN_SPANISH,
    speaker=enums.FrenchSpeakers.SPEAKER_3,
    digit=3,
)
```

which can then be visualized as follows:

```
visualization.plot_spectrogram(spgm)
```

See `spectrograms.py` for functionality to get transformed spectrograms (e.g., residual spectrograms).

### Covariance Analysis

This section of the paper takes each language's frequency covariance structure to be representative of how the language "sounds." We can get the frequency covariance structures for American Spanish and French with the following function calls, which return NumPy arrays:

```
spanish_cov = covariance_analysis.get_freq_covariance(
    language_family=enums.RomanceLanguages,
    language=enums.RomanceLanguages.AMERICAN_SPANISH
)

french_cov = covariance_analysis.get_freq_covariance(
    language_family=enums.RomanceLanguages,
    language=enums.RomanceLanguages.FRENCH
)
```

and then visualize them:

```
visualization.plot_freq_covariance(spanish_cov)

visualization.plot_freq_covariance(french_cov)
```

We can also find the distance between these two covariance structures (i.e., between the two languages' "sounds" more broadly):

```
covariance_analysis.get_procrustes_distance_svd(
    langfam=enums.RomanceLanguages,
    lang1=enums.RomanceLanguages.AMERICAN_SPANISH,
    lang2=enums.RomanceLanguages.FRENCH,
    cov_type=enums.CovarianceType.FREQUENCY
)
```

These pair-wise distances between languages are the basis for the paper's notion of a *language space*.

What's more, we can get the interpolated spectrograms between an American Spanish speaker (e.g., American Spanish speaker \#3) and a French speaker (e.g., French speaker \#8) saying the words for some digit (e.g., "3") in their respective languages:

```
covariance_analysis.interspeaker_interp(
    langfam=enums.RomanceLanguages,
    lang1=enums.RomanceLanguages.AMERICAN_SPANISH,
    lang2=enums.RomanceLanguages.FRENCH,
    speaker1=enums.AmericanSpanishSpeakers.SPEAKER_3,
    speaker2=enums.FrenchSpeakers.SPEAKER_8,
    digit=3,
)
```

These spectrograms (including the interpolations between them) are the basis for the paper's notion of *speaker space*.

### Hyperbolic Analysis

This paper proposes methods that embed spectrograms in hyperbolic space to perform the same types of analysis as that in the above section. It uses Poincaré disks – 2D representations of hyperbolic space – to represent both language and speaker space.  

The hyperbolic speaker space for a given digit is a Poincaré disk whose points are the embedded spectrograms of speakers (from all languages) saying a given digit. We get these embeddings by first constructing the corresponding $k$-nearest neighbor graph (e.g., for pronounciations of words for "3" in the Romance family using $k = 6$):

```
G = hyperbolic_analysis.construct_knn_graph(
    language_family=enums.RomanceLanguages,
    digit=3,
    k=6
)
```

and then computing the Poincaré embeddings for the speaker space based off of this graph:

```
X = hyperbolic_analysis.get_embeddings(
    graph=G,
    language_family=enums.RomanceLanguages
)
```

which we can then visualize on the Poincaré disk:

```
visualization.plot_poincare_disk(
    embedding_dict=X,
    digit=3,
    k=6,
    language_family=enums.RomanceLanguages
)
```

Suppose we wanted to interpolate between two speakers in this space (for instance, American Spanish speaker \#3 and French speaker \#8). The same workflow applies, but with an additional `interp_settings` parameter:

```
interp_settings = hyperbolic_analysis.InterpSettings(
    lang1=enums.RomanceLanguages.AMERICAN_SPANISH,
    lang2=enums.RomanceLanguages.FRENCH,
    speaker1=enums.AmericanSpanishSpeakers.SPEAKER_3,
    speaker2=enums.FrenchSpeakers.SPEAKER_8,
    digit=3,
)

G_interp = hyperbolic_analysis.construct_knn_graph(
    language_family=enums.RomanceLanguages,
    digit=3,
    k=6,
    interp_settings=interp_settings
)

X_interp = hyperbolic_analysis.get_embeddings(
    graph=G_interp,
    language_family=enums.RomanceLanguages,   
)

visualization.plot_poincare_disk(
    embedding_dict=X_interp,
    digit=3,
    k=6,
    language_family=enums.RomanceLanguages,
    interp_settings=interp_settings
)
```

We can also visualize the entire language space (e.g., that of the Romance languages) for a given value of $k$ in the hyperbolic setting by aggregating over all digit-wise disks:

```
X = hyperbolic_analysis.align_all_digit_disks(
    k=6,
    language_family=enums.RomanceLanguages
)

visualization.plot_aligned_digit_centroids(
    embedding_dict=X,
    k=6,
    language_family=enums.RomanceLanguages,
    radii=True,
)
```

<br>

## Citation and Reference

<mark>***TODO: change citation to reflect publishing status of paper as we go***</mark>

If you find this repo useful in your own research, please cite it with the following reference:

```
@article{hyperbolic-phonetics,
  title={Analyzing Language Families via Hyperbolic Embeddings of Phonetic Data},
  author={McDonald, Patrick and Weber, Melanie},
  journal={Under Review},
  year={2024}
}
```


[1]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[2]: https://en.wikipedia.org/wiki/Spectrogram