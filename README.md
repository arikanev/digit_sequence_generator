# Digit Sequence Generator

## generate_numbers_sequence.py
A script and API for generating a sequence of mnist digits based on a specified input list of digits.

To run as a script:

`python generate_numbers_sequence.py -d DIGITS (space-separated ints) -r SPACING_RANGE (two space-separated ints) -w IMAGE_WIDTH (int)` optional arguments: `-a AUGMENTATION (str)`

Currently the only augmentation supported is 'mnistm', which consists of mnist masks super-imposed on imagenet image backgrounds. A full mnistm dataset can be found [here](http://akanev.com/datasets).

To call as an API in any python code:

```
import generate_numbers_sequence

sequence = generate_numbers_sequence.generate_numbers_sequence([digit_list], (range_tuple), width_int)
```

Note, as of now you can only access augmentation options when running generate_numbers_sequence as a script.

The purpose of this code is to aid in training classifiers and generative deep-learning models. It is a semi-vectorized implementation where time is saved by computing an idx_mask for all mnist images at one time, and space is saved by recomputing the idx_mask for every input sequence digit.

To test the above API and script, run `python run_tests.py`.

Future work could involve extending the code to use any digit database for random image sampling.
