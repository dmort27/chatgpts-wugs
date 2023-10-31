# Affix Rule Learner

To run on a given directory (containing a `.trn` file for training data and 
a `.nonce` file for test data), run `python arl.py --path path/to/dir/`). The 
optional flag `--output` writes predictions to `.out` files. The optional flag
`--num k` uses the top `k` guesses instead of making only one prediction.
