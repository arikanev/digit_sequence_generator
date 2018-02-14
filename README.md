# Digit Sequence Generator

## generate_numbers_sequence.py

A script and API for generating and augmenting a sequence of digits based on a specified input list of digits. The purpose of this code is to aid in training classifiers and generative deep-learning models.

Running `generate_numbers_sequence.py` with augmentation saves a pair of image sequences with the following filenames:

* 'sequenceX.png'
* 'aug_sequenceX.png'

*(Where X is an integer denoting number of existing sequence files + 1)*

<br><br>
These files contain the exact same digit images in their sequences, and differ only by RGB and Greyscale value.

`generate_numbers_sequence.py` is a semi-vectorized implementation:

* Time is saved by vectorizing the method to generate an image.
* Space is saved by recomputing the method to generate an image for every digit in an input list.

<br><br>
### To run as a script:

```
python generate_numbers_sequence.py -d DIGITS (space-separated ints) -r SPACING_RANGE (two space-separated ints) -w IMAGE_WIDTH (int)
```
optional arguments:
```
-a AUGMENTATION (str)
```

Currently the augmentation supported is 'mnistm', which consists of mnist masks super-imposed on imagenet image backgrounds. A full mnistm dataset can be found [here](http://akanev.com/datasets).

*(As of now you can only access augmentation options when running generate_numbers_sequence as a script.)*

<br><br>
### To call as an API in python code:

```
import generate_numbers_sequence

sequence = generate_numbers_sequence.generate_numbers_sequence([digit_list], (range_tuple), width_int)
```

`generate_numbers_sequence.generate_numbers_sequence([digit_list], (range_tuple), width_int)` will return a numpy array of size `(height, width_int)` and dtype `float32`.

*(The height value is translated unchanged from the sampled digit image height)*

<br><br>
### Testing

To test the above API and script, run `python run_tests.py`.

In the future, tests should be added to: 

* Assert the shape match between sequenceX.png and aug_sequenceX.png.
* Ensure lack of runtime errors when generating images from other datasets.
*(For generalizability/extensability)*

<br><br>
### Future plans
Could/Should focus on:
  
* Expanding on the number of augmentation methods.
* Adding an option for sequence margins to be extended, as opposed to stretching the entire image.
* More
