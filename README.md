# CSCI 8360 - Project 3 - Neuron Finding

This project is implemented as a part of the Data Science Practicum (CSCI 8360) course at the University of Georgia, Spring 2019.
The goal was to develop an image segmentation pipeline that identifies as many of the neurons present as possible, as accurately as possible.

Please refer [Wiki](https://github.com/dsp-uga/Team-keller/wiki) for more details on our approach.

## Getting Started 

The following instructions will assist you get this project running on your local machine for developing and testing purpose.

### Prerequisites:

1. Python: <br />
    To install Python, go [here](https://www.python.org/downloads/)
    
2. Tensorflow: <br />
    If you don't have it installed, [download Tensorflow here](https://www.tensorflow.org/install).

3. Opencv: <br />
    `pip install opencv-python` 

4. [Google cloud platform](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Google-Cloud-Platform-set-up)

### Data Preparation:
The training and testing data folders are available on GCP bucket: `gs://uga-dsp/project3` <br />
Training datasets are provided with ground truth labeled regions for identified neurons, and testing datasets are provided without ground truth. Each downloadable dataset includes metadata (as `JSON`), images (as `TIFF`), and coordinates of identified neurons, also known as ROIs (as `JSON`). Datasets are around 1 GB zipped and a few GBs unzipped. Visit the [neurofinder](https://github.com/codeneuro/neurofinder#datasets) repository for current download links for all datasets.

Download these files into your project directory using gsutil:<br />
`gsutil cp -r gs://uga-dsp/project2/* base_dir`

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


### Model Approach:

- [NMF](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)
- [Tiramisu](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Future-Work)

## Ethics Considerations
This project can be used as a part of a bigger project to detect and identify the effects of Cilia movement/changes that can be used for future Medical Research. This project was trained on Medical images and should only be used for detecting cilia movement from the video pipeline. 

[LICENSE](https://github.com/dsp-uga/Team-keller/blob/master/ETHICS)

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

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

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/Team-thweatt-p2/blob/master/LICENSE) file for details

