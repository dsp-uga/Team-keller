# CSCI 8360 - Project 3 - Neuron Finding
[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

This project is implemented as a part of the Data Science Practicum (CSCI 8360) course at the University of Georgia, Spring 2019.
The goal was to develop an image segmentation pipeline that identifies as many of the neurons present as possible, as accurately as possible.

Please refer [Wiki](https://github.com/dsp-uga/Team-keller/wiki) for more details on our approach.

## Getting Started 

The following instructions will assist you get this project running on any machine for developing and testing purpose.

### Prerequisites:

1. Python: <br />
    To install Python, go [here](https://www.python.org/downloads/)
    
2. PyTorch: <br />
    To install PyTorch, use
    `pip3 install torch torchvision`
    For more information, visit the [PyTorch website](https://pytorch.org/).

3. [Thunder](https://github.com/thunder-project/thunder): <br />
    `pip install thunder-python` <br />
    `pip install thunder-extraction` <br />
    
4. [Google cloud platform](https://cloud.google.com)

### Data Preparation:
The training and testing data folders are available on GCP bucket: `gs://uga-dsp/project3` <br />
Training datasets are provided with ground truth labeled regions for identified neurons, and testing datasets are provided without ground truth. Each downloadable dataset includes metadata (as `JSON`), images (as `TIFF`), and coordinates of identified neurons, also known as ROIs (as `JSON`). Datasets are around 1 GB zipped and a few GBs unzipped. Visit the [neurofinder](https://github.com/codeneuro/neurofinder#datasets) repository for current download links for all datasets.

Download these files into your project directory using gsutil:<br />
`gsutil cp -r gs://uga-dsp/project3/* base_dir`

### Run Instruction:

You can download the source code and simply run the following command:

`$ python3 main.py --base_dir /path/to/project/directory/`

List of command line arguments to pass to the program are as follows:

	--base_dir: absolute project directory path.
	--clf: type of classifier to use. Current choices are 'rf' and 'svm'.
	--xdim: width of the images after preprocessing.
	--ydim: length of the images after preprocessing.
	--n_frames: number of the frames per video to consider.
	--pre_type: type of preprocessing. Choices are 'none', 'resize', or 'zero'.

The only reqired argument is the `base_dir` which is the directory containing `train_file`, `test_file`, `data\`, and `masks\`.

Then see the above list in command line to execute the following command:

`$ python3 main.py -h`

One typical usage is:

`$ python3 main.py --base_dir="../dataset/" --clf="rf" --xdim=640 --ydim=640 --n_frames=30 --pre_type="none"`

## Ethics Considerations
This project can be used as a part of a bigger study on the efficacy of new drugs on inhibiting certain types of cross-synaptic activity for the treatment of neurological disorders. With this context in mind, we have undertaken certain ethics considerations to ensure that this project cannot be misused for purposes other than the ones intended.

See the [ETHICS.md](https://github.com/dsp-uga/team-keller/blob/master/ETHICS.md) file for details.
Also see the [Wiki Ethics page](https://github.com/dsp-uga/Team-keller/wiki/Ethics) for explanations about the ethics considerations.

## Contributing

The `master` branch of this repo is write-protected and every pull request must passes a code review before being merged.
Other than that, there are no specific guidelines for contributing.
If you see something that can be improved, please send us a pull request!

## Authors
(Ordered alphabetically)

- **Dhaval Bhanderi**
- **Hemanth Dandu**
- **Sumer Singh**
- **Yang Shi** 

See the [CONTRIBUTORS.md](https://github.com/dsp-uga/team-keller/blob/master/CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/Team-keller/blob/master/LICENSE) file for details

