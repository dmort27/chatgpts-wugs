
## Modules to be installed

  * regex
  * Levenshtein
  * asyncio
  * Tensorflow

## Directories

The `eng`, `deu`, `tur`, and `tam` datasets are located in the `data/` directory.

The raw and IPA converted data files are located in the `data` directory in the root folder of the repository

The annotated files are located in the `data/{lang}/patterns` directory

The intermediate files are located in the `data/{lang}/steps` directory

To generate the annotated and intermediate files, run sigmorphon-2021-phono.ipynb

## Usage

Create `logs`, `models`, and `results` folders.

To execute the AED baseline code:

```
cd ./src
bash run_AED.sh
```



## References

We used inflectional FAPs in the experiments described in :

> Calderone, B., Hathout, N., Bonami, O., (2021). [Not quite there yet: Combining analogical patterns and encoder-decoder networks for cognitively plausible inflection.](https://aclanthology.org/2021.sigmorphon-1.28/) In *Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology* pp. 196-204. Bangkok, Thailand

The actual computation of FAPs is described in :

> Hathout, N., Sajous, F., Calderone, B., Namer, F. (2020). [Glawinette: a linguistically motivated derivational description of French acquired from GLAWI.](https://aclanthology.org/2020.lrec-1.478/) In *Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)* pp. 3870-3878. Marseille.