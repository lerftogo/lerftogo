# LERF-TOGO: Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping
This is the official implementation for [LERF-TOGO](https://lerftogo.github.io/).

# Installation

### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/lerftogo/lerftogo.git`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`
### 3. Install dependencies
```bash
cd robot_lerf/graspnet_baseline/knn
python setup.py install 
cd ../pointnet2
python setup.py
cd ../graspnetAPI
pip install -e .
```

# Using LERF-TOGO
Have your data in a folder named `data`
Run `python scripts/gen_grasp.py`
Enter data folder name to train/load data
Enter object;part query in lerf query text box
Generate grasps

## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{lerftogo2023,
&nbsp;title={Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping},
&nbsp;author={Adam Rashid and Satvik Sharma and Chung Min Kim and Justin Kerr and Lawrence Yunliang Chen and Angjoo Kanazawa and Ken Goldberg},
&nbsp;booktitle={7th Annual Conference on Robot Learning},
&nbsp;year = {2023},
} </pre>