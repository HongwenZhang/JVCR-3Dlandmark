# Joint Voxel and Coordinate Regression (JVCR) for 3D Facial Landmark Localization

This repository includes the PyTorch code of the JVCR method described in [Adversarial Learning Semantic Volume for 2D/3D Face Shape Regression in the Wild](https://openreview.net/pdf?id=gafjGfv8uR) (IEEE Transactions on Image Processing, 2019).

<p align='center'>
<img src='imgs/aflwDemo.gif' title='examples for 3D facial landmark localization' style='max-width:600px'></img>
</p>

## Requirements

- python 2.7

### packages

- [PyTorch](https://www.pytorch.org)
- [NumPy](http://www.numpy.org)
- [Matplotlib](https://matplotlib.org)
- [progress](https://anaconda.org/conda-forge/progress)

## Usage

Clone the repository and install the dependencies mentioned above
```
git clone https://github.com/HongwenZhang/JVCR-3Dlandmark.git
cd JVCR-3Dlandmark
```
Then, you can run the demo code or train a model from stratch.

### Demo
1. Download the [pre-trained model](https://drive.google.com/drive/folders/1wT3efHjqUfTMHj8qAjkPn8m9qS614Lxu) (trained on [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)) and put it into the `checkpoint` directory

2. Run the demo code

```
python run_demo.py --verbose
```

### Training

1. Prepare the training and evaluation datasets
- Download [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) and [AFLW3000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- Create soft links to the dataset directories
```
ln -s /path/to/your/300W_LP data/300wLP/images
ln -s /path/to/your/aflw2000 data/aflw2000/images
```
- Download `.json` annotation files from [here](https://drive.google.com/drive/folders/16cj4x1v1jbqikB4KS8ndnuP49coqwt1c) and put them into `data/300wLP` and `data/aflw2000` respectively
2. Run the training code
```
python train.py --gpus 0 -j 4
```

## Acknowledgment

The code is developed upon [PyTorch-Pose](https://github.com/bearpaw/pytorch-pose). Thanks to the original author.

## Citation
If the code is helpful in your research, please cite the following paper.
```
@article{zhang2019adversarial,
  title={Adversarial Learning Semantic Volume for 2D/3D Face Shape Regression in the Wild},
  author={Zhang, Hongwen and Li, Qi and Sun, Zhenan},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={9},
  pages={4526--4540},
  year={2019},
  publisher={IEEE}
}
```
