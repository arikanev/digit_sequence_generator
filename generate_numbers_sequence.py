"""This module generates a sequence of mnist images given a list of digits."""
import argparse
import glob
import numpy as np
import os
import pickle as pkl

from PIL import Image


def generate_numbers_sequence(digits, spacing_range, image_width):
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
    A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
    a (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        specifies the width of the image in pixels.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black)
    to 1 (white), the first dimension corresponding to the height and the
    second dimension to the width.

    Details
    -------
    This is a partially-vectorized implementation where we take a middle
    ground between efficiency in space vs time.

    Time is saved by computing idx_mask for all mnist images at once.

    Space is saved by recomputing idx_mask for each sequence digit.

    """
    data = load_pickle('mnist.pkl')

    # Pool train, test, val images.

    X, y = pool_data(data)

    # Image height equals sampled image(s) height.

    image_height = X[0].shape[0]

    # Initalize digit_sequence with (height, width) and fill with zeros.

    digit_sequence = np.zeros((image_height, sample_spacing(spacing_range)))

    # Iterate through digits.

    for digit in digits:

        # Sample an mnist digit.

        digit = sample_image(X, y, digit)

        # Generate a spacing.

        spacing = np.zeros((image_height, sample_spacing(spacing_range)))

        # Append digit to current sequence, and spacing to image.

        digit_sequence = np.hstack((digit_sequence, digit, spacing))

    # Alter width

    digit_sequence = alter_width(digit_sequence, image_height, image_width)

    # Return digit_sequence.

    return digit_sequence, image_height


def load_pickle(pklfile):
    """
    Use as helper method.

    Here we load in mnist from a .pkl file. Using pickle to store/load/
    transport data is a memory efficient method.

    An alternative but slower method of loading mnist would be iterating
    through a directory of mnist images.

    Due to a compatibility problem with numpy in 2.7 vs 3.x, we must specify
    an encoding for pkl.load() to read the mnist file properly.
    """
    return pkl.load(open(pklfile, 'rb'), encoding='latin1')


def pool_data(data):
    """
    Use as helper method.

    Here we stack train, test, val examples and labels to 'pool'
    all images together.
    """
    X = np.vstack((data[0][0], data[1][0], data[2][0]))

    # Reshape images to fit a (height, width) construct.

    h = w = int(np.sqrt(X[0].shape[0]))

    X = X.reshape(len(X), h, w)

    # Horizontally append labels.

    y = np.hstack((data[0][1], data[1][1], data[2][1]))

    return X, y


def sample_spacing(spacing_range):
    """
    Use as helper method.

    Here we sample a random image spacing from a range.

    np.random.random_integers is [low, high] (inclusive),
    as opposed to np.random.randint's [low, high).
    """
    return np.random.random_integers(spacing_range[0], spacing_range[1])


def sample_image(X, y, digit):
    """
    Use as helper method.

    idx_mask: numpy array of boolean values, specifying if
    mnist image at idx_mask matches the current digit in the sequence.
    """
    idx_mask = (y == np.full((len(X)), digit))

    # Extract relevant images.

    pool = X[idx_mask]

    """
    Trade-off on random selection
    -----------------------------
    Rather than another dependancy, we use numpy and a more complex line.
    """

    # Sample random mnist image from pool.

    return pool[np.random.randint(len(pool))]


def numpy2im(digit_sequence):
    """
    Use as helper method.

    Converts a numpy array to a PIL Image object.
    """
    return Image.fromarray(np.uint8(digit_sequence * 255))


def alter_width(digit_sequence, image_height, image_width):
    """
    Use as helper method.

    Alters the width of a numpy array.
    Uses interpolation method from PIL Image.
    """
    # Convert numpy array to PIL Image object.

    digit_sequence = numpy2im(digit_sequence)

    # Resize generated image width by image_width pixels.

    digit_sequence = digit_sequence.resize((image_width, image_height),
                                           Image.ANTIALIAS)

    # Convert PIL Image object to numpy array.

    digit_sequence = np.array(digit_sequence).astype('float32') / 255

    return digit_sequence


def save_sequence(digit_sequence, aug=False):
    """
    Use as helper method.

    Saves a PIL Image object to the current directory as unique file.
    """
    if (aug):

        # Count number of generated sequence files minus one.

        num_sequence_files = len(glob.glob1(os.getcwd(), 'sequence*')) - 1

        digit_sequence.save('aug_sequence{}.png'.format(num_sequence_files))
        print("Saved aug_sequence{}.png".format(num_sequence_files))

    else:

        # Count number of generated sequence files.

        num_sequence_files = len(glob.glob1(os.getcwd(), 'sequence*'))

        digit_sequence.save('sequence{}.png'.format(num_sequence_files))
        print("Saved sequence{}.png".format(num_sequence_files))


def parse_args():
    """
    Argparse is used here for command-line compatibility/script purposes.

    We're able to load parameters input from command-line into
    generate_numbers_sequence() with the below argparse code.

    An alternative method for reading command-line inputs would be
    sys.argv[idx].
    """
    parser = argparse.ArgumentParser()

    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')

    optional = parser.add_argument_group('optional arguments')

    """
    Required arguments for generate_numbers_sequence().
    """

    required.add_argument('-d', '--digits', nargs='+', type=int, required=True,
                          help='List of digits from which a sequence will be \
                          generated')

    required.add_argument('-r', '--spacing_range', nargs=2, type=int,
                          required=True, help='Min and Max pixel range between \
                          images')

    required.add_argument('-w', '--image_width', type=int, required=True,
                          help='Final width of generated image')

    """
    Optional arguments for generate_numbers_sequence(). These args have default
    values.
    """

    optional.add_argument('-a', '--augmentation', type=str, default='none',
                          help='Type of augmentation on the generated image \
                          sequence')

    # Parse arguments.

    args = parser.parse_args()

    return args


def augment_sequence(greyscale_sequence, image_height):
    """
    Use as helper method.

    Here we have the possibility to augment the sequence in
    the form of mnistm images.

    A full mnistm set can be found at akanev.com/datasets.
    """
    background = load_pickle('imagenet.pkl')

    # Get a random imagenet background from array.

    background = background[np.random.randint(len(background))]

    Image.fromarray(background).save('before_alter_width.png')

    # Alter the width.

    background = np.uint8(alter_width(background, image_height,
                                      args.image_width) * 255)

    # Convert greyscale sequence to RGB for compatibility.

    digit_sequence = np.empty((image_height, args.image_width, 3),
                              dtype=np.uint8)

    digit_sequence[:, :, 0] = np.uint8(greyscale_sequence * 255)
    digit_sequence[:, :, 1] = digit_sequence[:, :, 0]
    digit_sequence[:, :, 2] = digit_sequence[:, :, 1]

    # Set sequence values over & under a threshold to imagenet values.

    digit_sequence[digit_sequence < 127] = background[digit_sequence < 127]
    digit_sequence[digit_sequence > 126] = 255 - background[digit_sequence >
                                                            126]

    # Convert numpy array to PIL Image object and return.

    return Image.fromarray(digit_sequence)


if __name__ == '__main__':

    # Parse command line arguments.

    args = parse_args()

    # Create a tuple from min and max spacing range.

    spacing_range = (args.spacing_range[0], args.spacing_range[1])

    # Generate an image sequence, get image_height.

    digit_sequence, image_height = generate_numbers_sequence(digits=args.digits,
                                                             spacing_range=spacing_range,
                                                             image_width=args.image_width)

    # Save generated sequence to file.

    save_sequence(numpy2im(digit_sequence), aug=False)

    # Use if 'mnistm' is specified in the arg 'augmentation'.

    if (args.augmentation == 'mnistm'):

        # Augment sequence

        augmented_sequence = augment_sequence(digit_sequence, image_height)

        # Save generated sequence to file.

        save_sequence(augmented_sequence, aug=True)
