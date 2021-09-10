# MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention
Official implementation of MC Denoising via Auxiliary Feature Guided Self-Attention (SIGGRAPH Asia 2021 paper) \[[PDF](http://nieyongwei.net/download/SigAsia_2021_Monte_Carlo_Denoising_low_res.pdf)\]

## Notice
* We now have the time and resources to train a model with diffuse and specular decomposition.
For a fairer comparison with previous models (w/ diffuse and specular decomposition), its model weights, codes and evaluation results will be published soon!
* Code implemented in [Jittor](https://github.com/Jittor/jittor) (a Just-in-time(JIT) deep learning framework) will be published soon!
* We trained all our models on Nvidia RTX 3090. We have tried to use automatic mixed precision (amp) in training to reduce memory usage and speed up the process, unfortunately, it inevitably causes NaN loss. 
* Instead, we used gradient checkpoints on some of the Transformer blocks to reduce memory usage (with the downside of increasing training time a bit).

## Abstract
While self-attention has been successfully applied in a variety of natural language processing and computer vision tasks, its application in Monte Carlo (MC) image denoising has not yet been well explored. This paper presents a self-attention based MC denoising deep learning network based on the fact that self-attention is essentially non-local means filtering in the embedding space which makes it inherently very suitable for the denoising task. Particularly, we modify the standard self-attention mechanism to an auxiliary feature guided self-attention that considers the by-products (e.g., auxiliary feature buffers) of the MC rendering process. As a critical prerequisite to fully exploit the performance of self-attention, we design a multi-scale feature extraction stage, which provides a rich set of raw features for the later self-attention module. As self-attention poses a high computational complexity, we describe several ways that accelerate it. Ablation experiments validate the necessity and effectiveness of the above design choices. Comparison experiments show that the proposed self-attention based MC denoising method outperforms the current state-of-the-art methods. 

![Teaser](https://github.com/Aatr0x13/MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention/blob/main/assets/teaser.png)

#### Fig. 1. Current state-of-the-art methods, including NFOR \[Bitterli et al.2016], KPCN \[Bako et al.2017] and ACFM \[Xu et al.2019], fail to produce a plausible denoised image for the scene "VeachAjar" because of the absence of specular albedo and the extremely noisy input. In contrast, our proposed model with auxiliary feature guided self-attention can gather the most relevant information for each pixel from its surrounding region in an edge-preserving manner, thus better restoring image details while preserving image structures and producing visually pleasing denoising results.

## Dependencies (other versions may also work)
* Python 3.8
* PyTorch 1.8.0
* [pyexr](https://github.com/tvogels/pyexr) 0.3.9
* [prefetch_generator](https://github.com/justheuristic/prefetch_generator) 1.0.1
* [h5py](https://github.com/h5py/h5py) 2.10.0

## Dataset
* The training set is provided by ACFM \[Xu et al. 2019], which contains a total of 1109 [Tungsten](https://github.com/tunabrain/tungsten) shots with the noisy images rendered at 32spp and the gt images rendered at 32768spp.
* The test set contains 14 shots of the default Tungsten scene with the noise images rendered at 4spp, 8spp, 16spp, 32spp, 128spp, 256spp, 512spp and 1024spp, while the gt images are shipped with the scene files. All these images are divided into patches of size 128 x 128 and stored as .h5 files. These files can be downloaded from [googledrive]().

## Model weights
* Model weights w/o diffuse and specular decomposition can be downloaded from [googledrive](https://drive.google.com/file/d/12iyOwhxdqoHwNtQ9Y01FEttYG30jVPrQ/view?usp=sharing)

  Please put the weights in <kbd>models\wo_diff_spec_decomp</kbd>.
* Model weights w/ diffuse and specular decomposition will be published soon!

  Please put the weights in <kbd>models\w_diff_spec_decomp</kbd>.

## Train and evaluate
For more options, please refer to the code.
* To train, run:
```
python train.py -i PATH_TO_IMAGE_FOLDER -d PATH_TO_TRAIN_H5_FOLDER -o PATH_TO_OUTPUT_FOLDER
```
*Example (with subfolders <kbd>32spp</kbd> and <kbd>32768spp</kbd> in the folder <kbd>image</kbd> and storing the dataset .h5 file in the folder <kbd>dataset</kbd>):
```
python train.py -i image -d dataset -o output
```  
***
* To test (.h5), run:
```
python test.py -i PATH_TO_TEST_H5_FOLDER -o PATH_TO_OUTPUT_FOLDER
```
*Example (with <kbd>test_4.h5</kbd> in the folder <kbd>h5</kbd>):
```
python test.py -i h5 -o output --datasetSPP 4
```  
***  
* To inference (full image), run:
```
python inference.py -i PATH_TO_IMAGE_FOLDER -o PATH_TO_OUTPUT_FOLDER --fileName NAME_OF_IAMGE
```
*Example (with <kbd>veach-ajar.exr</kbd> and <kbd>veach-ajar_gt.exr</kbd> in the folder <kbd>image</kbd>):
```
python inference.py -i image -o output --fileName veach-ajar --isLoadGt
```

## Results
|      |      |
|:----:|:----:|
|      |      |
|      |      |

## Citation
If you find our work useful in your research, please consider citing:
```
```

## Acknowledgments
Some of our code is adapted/ported from [KPCN](https://github.com/Nidjo123/kpcn) (implemented in PyTorch) and [ACFM](https://github.com/mcdenoising/AdvMCDenoise). Credit to these PyTorch projects.
