# MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention
Official implementation of MC Denoising via Auxiliary Feature Guided Self-Attention (SIGGRAPH Asia 2021 paper)
[PDF]()

## Notes
* We now have the time and resources to train a model with diffuse and specular decomposition.
For a fairer comparison with previous models (w/ diffuse and specular decomposition), its model weights, codes and evaluation results will be published soon!
* Codes implemented in [Jittor](https://github.com/Jittor/jittor) (a Just-in-time(JIT) deep learning framework) will be published soon!
* We trained all our models on Nvidia RTX 3090. We have tried to use automatic mixed precision (amp) in training to reduce memory usage and speed up the process, unfortunately, it inevitably causes NaN loss. 
* Instead, we used gradient checkpoints on some of the Transformer blocks to reduce memory usage (with the downside of increasing training time a bit).

## Abstract
While self-attention has been successfully applied in a variety of natural language processing and computer vision tasks, its application in Monte Carlo (MC) image denoising has not yet been well explored. This paper presents a self-attention based MC denoising deep learning network based on the fact that self-attention is essentially non-local means filtering in the embedding space which makes it inherently very suitable for the denoising task. Particularly, we modify the standard self-attention mechanism to an auxiliary feature guided self-attention that considers the by-products (e.g., auxiliary feature buffers) of the MC rendering process. As a critical prerequisite to fully exploit the performance of self-attention, we design a multi-scale feature extraction stage, which provides a rich set of raw features for the later self-attention module. As self-attention poses a high computational complexity, we describe several ways that accelerate it. Ablation experiments validate the necessity and effectiveness of the above design choices. Comparison experiments show that the proposed self-attention based MC denoising method outperforms the current state-of-the-art methods. 

## Dependencies
* Python 3.8
* PyTorch 1.8.0
* [pyexr](https://github.com/tvogels/pyexr) 0.3.9
* [prefetch_generator](https://github.com/justheuristic/prefetch_generator) 1.0.1

## Dataset

## Model weights
* Model weights w/o diffuse and specular decomposition can be downloaded from [googledrive](https://drive.google.com/file/d/12iyOwhxdqoHwNtQ9Y01FEttYG30jVPrQ/view?usp=sharing)
* Model weights w/ diffuse and specular decomposition will be published soon!

## Train and evaluate

## Results

## Citation

## Acknowledgments
