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
    """
    Initialize digit_sequence with:

        elements: 0 (black)
        height: 28
        width: int randomly selected from spacing_range.

    np.random.random_integers is [low, high] (inclusive),
     as opposed to np.random.randint's [low, high).
    """
    spacing = np.random.random_integers(spacing_range[0], spacing_range[1])
    digit_sequence = np.zeros((28, spacing))

    """
    Here we load in mnist from a .pkl file. Using pickle to store/load/
    transport data is a memory efficient method. An alternative but
    considerably slower method of loading mnist would be iterating
    through a directory of mnist images.

    Due to a compatibility problem with numpy in 2.7 vs 3.x, we must specify
    and encoding for pkl.load() to read the mnist file properly.
    """

    data = pkl.load(open('mnist.pkl', 'rb'), encoding='latin1')

    """
    Here we stack train, test, val examples and labels to 'pool'
    all images together.
    """

    # Vertically stack images.

    X = np.vstack((data[0][0], data[1][0], data[2][0]))

    # Reshape images to fit a (height, width) construct.

    X = X.reshape(70000, 28, 28)

    # Horizontally append labels.

    y = np.hstack((data[0][1], data[1][1], data[2][1]))

    for digit in digits:

        """
        idx_mask: numpy array of boolean values, specifying if
        mnist image at idx_mask matches the current digit in the sequence.
        """
        # Get idx_mask for current digit.

        idx_mask = (y == np.full((70000), digit))

        # Extract relevant images.

        pool = X[idx_mask]

        """
        Trade-off on random selection
        -----------------------------
        Rather than another dependancy, we use numpy and a more complex line.
        """

        # Sample random mnist image from pool.

        image = pool[np.random.randint(len(pool))]

        # Generate a spacing filled with 0 (black values)
        spacing = np.random.random_integers(spacing_range[0], spacing_range[1])
        spacing = np.zeros((28, spacing))

        # Append image to digit_sequence, and 0 (black) values to image.

        digit_sequence = np.hstack((digit_sequence, image, spacing))

    # Convert numpy array to PIL Image object.

    image = Image.fromarray(np.uint8(digit_sequence * 255))

    # Resize generated image width by image_width pixels.

    image = image.resize((image_width, 28), Image.ANTIALIAS)

    # Convert PIL Image object to numpy array.

    digit_sequence = np.array(image).astype('float32') / 255

    # Return digit_sequence.

    return digit_sequence


if __name__ == '__main__':
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

    # Create a tuple from min and max spacing range.

    spacing_range = (args.spacing_range[0], args.spacing_range[1])

    # Generate an image sequence.

    sequence_array = generate_numbers_sequence(digits=args.digits,
                                               spacing_range=spacing_range,
                                               image_width=args.image_width)

    # Rescale and convert numpy array to PIL Image object.

    sequence = Image.fromarray(np.uint8(sequence_array * 255))

    # Count number of generated sequence files.

    num_sequence_files = len(glob.glob1(os.getcwd(), 'sequence*'))

    # Save generated sequence as unique file.

    sequence.save('sequence{}.png'.format(num_sequence_files))
    print("Saved sequence{}.png".format(num_sequence_files))

    """
    Here we have the possibility to augment the sequence in
    the form of mnistm images.

    A full mnistm set can be found at akanev.com/datasets.
    """

    # Use if 'mnistm' is specified in the argparse 'augmentation'.

    if (args.augmentation == 'mnistm'):

        # Open the imagenet pickle file.

        with open('imagenet.pkl', 'rb') as infile:

            # Read pickle file.

            imgnet_array = pkl.load(infile)

            # Get a random imgnet_img from array.

            imgnet_img = imgnet_array[np.random.randint(len(imgnet_array))]

            # Convert imgnet_img to PIL Image object.

            imgnet_img = Image.fromarray(imgnet_img)

            # Resize PIL Image object to equal sequence width and height.

            imgnet_img = imgnet_img.resize((args.image_width, 28),
                                           Image.ANTIALIAS)

            # Convert PIL Image object back to numpy array.

            imgnet_img = np.array(imgnet_img)

            # Convert greyscale sequence to RGB for compatibility.

            rgb_sequence = np.empty((28, args.image_width, 3), dtype=np.uint8)
            rgb_sequence[:, :, 0] = np.uint8(sequence_array * 255)
            rgb_sequence[:, :, 1] = rgb_sequence[:, :, 0]
            rgb_sequence[:, :, 2] = rgb_sequence[:, :, 1]

            # Set sequence values over & under a threshold to imagenet values.

            rgb_sequence[rgb_sequence < 127] = imgnet_img[rgb_sequence < 127]
            rgb_sequence[rgb_sequence > 126] = 255 - imgnet_img[rgb_sequence >
                                                                126]

            # Convert numpy array to PIL Image object

            rgb_sequence = Image.fromarray(rgb_sequence)

            # Save generated sequence as unique file.

            rgb_sequence.save('aug_sequence{}.png'.format(num_sequence_files))
            print("Saved aug_sequence{}.png".format(num_sequence_files))
