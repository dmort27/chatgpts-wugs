# Introduction

The nonce data for this project are distributed as an encrypted 7-zip archive. The purpose behind this decision is to prevent them from contaminating future scrapes of GitHub and therefore becoming part of the training data of future NLP models. That would destroy their usefulness as a benchmark data set. At the same time, we want to make the data set available to the community.

# 7-zip

[7-zip](https://www.7-zip.org/) is a cross-platform, open source file compression program that supports AES-256 encryption. In principle, we could have used password-protected ZIP archives (which use a weaker form of encryption) but we reasoned that some future web scrapers might include the ability to exploit the serious vulnerabilities in the zip encryption algorithm. We reasoned that this was less likely to be the case with AES-256 encryption.

# Extracting the Data

To extract the data, follow these steps:

- Obtain the password for the archive from David Mortensen at dmortens@cs.cmu.edu (or, failing that, davidmortensen@gmail.com)
- Install 7-zip
- Run the following at the commandline (Linux, Windows, or MacOS), mutatis mutandis:

        7z x chatgpts-wugs.7z
and supply the provided password. On MacOS, the 7-zip binary is usually called `7zz`.

# Training Data

We provide the train/dev/test data used in this project as they consist of only real words. Run `python ortho_to_IPA.py` with the desired arguments to convert orthographic data to IPA.

The English and German train/dev/test data are taken from the [SIGMORPHON 2023 Inflection Shared Task](https://github.com/sigmorphon/2023InflectionST).

The Tamil and Turkish train/dev/test data are generated by the authors.