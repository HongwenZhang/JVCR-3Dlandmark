# Joint Voxel and Coordinate Regression for Accurate 3D Facial Landmark Localization

This repository includes the PyTorch code for evaluation and visualization of the network described in [Joint Voxel and Coordinate Regression for Accurate 3D Facial Landmark Localization](https://arxiv.org/abs/1801.09242).

<p align='center'>
<img src='imgs/aflwDemo.gif' title='examples for 3D facial landmark localization' style='max-width:600px'></img>
</p>

## Requirements

- python 2.7

### packages

- [pytorch](https://www.pytorch.org)
- [numpy](http://www.numpy.org)
- [matplotlib](https://matplotlib.org)

## Usage

1. Clone the repository and install all the dependencies mentioned above
```
git clone https://github.com/HongwenZhang/JVCR-3Dlandmark.git
cd JVCR-3Dlandmark
```

2. Download the [pre-trained model](https://drive.google.com/drive/folders/1wT3efHjqUfTMHj8qAjkPn8m9qS614Lxu) (trained on [300wLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)) and put it into the `checkpoint` directory

3. Run the demo code

```
python run_demo.py --verbose
```

## Acknowledgment

The code is developed upon [PyTorch-Pose](https://github.com/bearpaw/pytorch-pose). Thanks to the original author.