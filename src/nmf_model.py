import json
import thunder as td
from extraction import NMF
import os
import argparse


def compute_nmf(
        datasets,
        var_data_path,
        var_num_components,
        var_percentile,
        var_max_iter,
        var_overlap,
        var_chunk_size):
    """
    This code is a modified version of the baseline code given below.
    @ https://gist.github.com/freeman-lab/330183fdb0ea7f4103deddc9fae18113/
    This function performs nmf and saves the results.

    Parameters
    ----------
    datasets : List
        List of dataset names.
    var_data_path: String
        Path to data folder
    var_num_components: Int
        The number of components to estimate per block.
    var_percentile: Int
        The value for thresholding.
    var_max_iter : Int
        The maximum number of algorithm iterations.
    var_overlap: Int
        The value for determining whether to merge.
    var_chunk_size: Int
        The the chunk size.
    """


    if var_data_path[-1] != '/':
        var_data_path = var_data_path + '/'

    submission = []

    for dataset in datasets:
        path = var_data_path + dataset
        data = td.images.fromtif(path + '/images', ext='tiff')
        print('done')
        algorithm = NMF(
            k=var_num_components,
            percentile=var_percentile,
            max_iter=var_max_iter,
            overlap=var_overlap)
        model = algorithm.fit(
            data, chunk_size=(
                var_chunk_size, var_chunk_size))
        print('done')
        merged = model.merge(var_overlap)
        print('done')
        print('found %g regions' % merged.regions.count)
        regions = [{'coordinates': region.coordinates.tolist()}
                   for region in merged.regions]
        # We slice dataset at 12 to reomvove 'neurofinder' from the name
        result = {'dataset': dataset[12:], 'regions': regions}
        submission.append(result)

        print('writing results')

        with open(var_data_path + 'submission.json', 'w') as f:
            f.write(json.dumps(submission))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Trains the model and outputs predictions.'),
        add_help='How to use', prog='nmf_model.py <args>')

    # Required arguments
    parser.add_argument("-d", "--data_path", required=True,
                        help=("Provide the path to the data folder"))

    # Optional arguments
    parser.add_argument(
        "-k",
        "--num_components",
        default=5,
        help=("Set the number of components to estimate per block"))

    parser.add_argument(
        "-p",
        "--percentile",
        default=95,
        help=("Set the value for thresholding (higher means more thresholding)"))

    parser.add_argument(
        "-m",
        "--max_iter",
        default=20,
        help=("Set the maximum number of algorithm iterations"))

    parser.add_argument(
        "-o",
        "--overlap",
        default=0.1,
        help=("Set the value for determining whether to merge (higher means fewer merges)"))

    parser.add_argument("-c", "--chunk_size", default=50,
                        help=("Set the chunk size"))

    args = vars(parser.parse_args())

    files = sorted(os.listdir(args['data_path']))

    compute_nmf(
        files,
        args['data_path'],
        int(args['num_components']),
        int(args['percentile']),
        int(args['max_iter']),
        int(args['overlap']),
        int(args['chunk_size']))
