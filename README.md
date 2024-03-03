# Offical Implementation of TeTriRF

TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint Video (CVPR2024)
[project page](https://wuminye.github.io/projects/TeTriRF/), [paper](https://arxiv.org/pdf/2312.06713.pdf).

This paper presents Temporal Tri-Plane Radiance Fields (TeTriRF), a novel technology that significantly reduces the storage size for Free-Viewpoint Video (FVV) while maintaining low-cost generation and rendering.
![TeTriRF](https://wuminye.github.io/projects/TeTriRF/static/images/exp5.png)


### Installation

You can follow its installation steps:

```
git clone https://github.com/wuminye/TeTriRF.git
cd TeTriRF
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.


Please install [FFmpeg](https://ffmpeg.org/) with libx265 support in your system.

## Dataset

This code supports  [NHR](https://github.com/wuminye/NHR), [ReRF](https://github.com/aoliao12138/ReRF_Dataset), and [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video) datasets.

For NHR and ReRF datasets, please use [nhr_conversion.py](tools/nhr_conversion.py) and [rerf_to_nhr_conversion.py](tools/rerf_to_nhr_conversion.py) respectively for data conversions.

For DyNeRF dataset, please use [n3d_llf.py](tools/n3d_llf.py) for data conversions.


## GO

- Training

    NHR and ReRF scenes:

    ```bash
    $ python gen_train.py
    ```

    DyNeRF scenes:
    ```bash
    $ python gen_n3d.py
    ```

    Please modify gen_train.py file to specify the config file.

- Evaluation
    
    Testing on uncompressed representations:
    ```bash
    $ python gen_test.py
    ```
    Rate-distortion curves evaluations (after compression and decompression):
    ```bash
    $ python gen_rate_distortion.py
    ```

    Do not forget to modify these python file to specify the config files or folder paths. We use 'wandb' to log results.

- Render video
    ```bash
    $ python gen_360.py
    ```
    Please modify gen_360.py file to specify the config file.


- Compression

    This command can compress and package the trained sequence into a zip file (in the specified output folder), containing the necessary information for playback.
    ```bash
    $ python tools/compression.py --logdir <path to the output folder> --numframe <number of frames> --qp <compression quality, smaller values mean larger storage and better quality., default:20 >
    ```

- Player
    WIP...




## Acknowledgement
The code base is origined from an awesome [DVGO](https://github.com/sunset1995/DirectVoxGO) implementation, but it becomes very different from the code base now.

If you're using TeTriRF in your research or applications, please cite using this BibTeX:
```bibtex
@article{wu2023tetrirf,
  title={TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint Video},
  author={Wu, Minye and Wang, Zehao and Kouros, Georgios and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:2312.06713},
  year={2023}
}
```