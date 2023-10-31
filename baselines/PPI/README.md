## This is the repository for the Principle Parts of Inflection (PPI) baseline


## Directories

The `eng`, `deu`, `tur`, and `tam` datasets are located in the `data` directory in the repository root folder.

## Usage

### Create the paradigms for the different languages

`python get_paradigms.py`

### Run PPI for different languages on the dev/test split

`bash runAll.sh`

## Evaluate results on the test/dev split

`python eval.py`

### Generate the inflections for the nonce words 
`bash generate_nonce.sh`


## References

Please visit the corresponding github repo as the main source for PPI

https://github.com/LINGuistLIU/principal_parts_for_inflection

