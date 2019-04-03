import argparse
import os
import imageio
import numpy as np


def increase_contrast(
        datasets,
        var_data_path,
        var_save_location,
        var_upper_bound,
        var_lower_bound):
    """
    The function increases image contrast.
    It clips value above the upper bound and
    below the lower bound. Then it scales the
    remaining values between 0 and 65535 (16 bit).
    @http://papers.nips.cc/paper/6138-automatic-neuron-detection-in-calcium-imaging-data-using-convolutional-networks.pdf

    Parameters
    ----------
    datasets : List
        List of dataset names.
    var_data_path: String
        Path to data folder
    var_save_location: String
        Path to output folder.
    var_upper_bound: Int
        The upper threshold for clipping value.
    var_lower_bound : List
        The lower threshold for clipping value.
    """

    if var_data_path[-1] != '/':
        var_data_path = var_data_path + '/'

    if var_save_location[-1] != '/':
        var_save_location = var_save_location + '/'

    for dataset in datasets:
        path = var_data_path + dataset + '/images/'
        save_path = var_save_location + dataset + '/images/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_list = sorted(os.listdir(path))
        for images in image_list:

            if images[0:5] != 'image':
                continue
            full_path = path + images
            f = imageio.imread(full_path)
            small = int(np.percentile(f, var_lower_bound, axis=(0, 1)))
            big = int(np.percentile(f, var_upper_bound, axis=(0, 1)))
            new_pic = np.clip(f, small, big)
            d_type = new_pic.dtype
            if d_type == 'uint16':
                scaler = 65535
            elif d_type == 'uint8':
                scaler = 255
            scaled_pic = ((new_pic -
                             new_pic.min()) *
                            (1 /
                             (new_pic.max() -
                              new_pic.min()) *
                             scaler).astype(d_type))
            imageio.imwrite(save_path + images, scaled_pic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Increases the image contrast.'),
        add_help='How to use', prog='contrast.py <args>')

    # Required arguments
    parser.add_argument("-d", "--data_path", required=True,
                        help=("Provide the path to the data folder"))

    # Optional arguments
    parser.add_argument(
        "-s",
        "--save_location",
        default=os.getcwd() + '/edited_data',
        help=("Set the location to save the data"))

    parser.add_argument(
        "-u",
        "--upper_bound",
        default=99,
        help=("Set the upper bound"))

    parser.add_argument(
        "-l",
        "--lower_bound",
        default=3,
        help=("Set the lower bound)"))

    args = vars(parser.parse_args())

    files = sorted(os.listdir(args['data_path']))

    if not os.path.exists(args['save_location']):
        os.makedirs(args['save_location'])

    increase_contrast(
        files,
        args['data_path'],
        args['save_location'],
        int(args['upper_bound']),
        int(args['lower_bound']))
