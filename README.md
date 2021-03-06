## Reproducing "Dentritic cortical microcircuits approximate the backpropagation algorithm" (Sacramento et al, NeurIPS 2018) results

The following repository aims at reproducing the results of [this paper](https://arxiv.org/abs/1810.11393), Fig. 1 (random input target association) and Fig. 2 (non-linear regression task).

## Package requirements

Run the following command lines to set the environment using conda:
```
conda create --name EP python=3.6
conda activate EP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch
```

## Commands to be run

These are examples of commands to be executed to reproduce:

+ Fig. 1 of the supplementary:
  ```
  python main.py --action figs1 --lr_ip 0.0002375 --lr_pi 0.0005 --noise 0.1
  ```

+ Fig. 1 of the main:
  ```
  python main.py --action fig1 --lr_ip 0.0011875 --lr_pi 0 --lr_pp 0.0005 0.0011875 --noise 0.1
  ```

+ Fig. 2 of the main:
  ```
  python main.py --action fig2 --lr_ip 0.0011875 --lr_pi 0.0059375 --lr_pp 0.0005 0.0011875 --noise 0.3 --freeze-feedback
  ```
