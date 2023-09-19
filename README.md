# FETCH: A Memory-Efficient Replay Approach for Continual Learning in Image Classification

![](https://www.tu-chemnitz.de/etit/proaut/en/research/rsrc/fetch/fetch.png)

This repository contains the source code for our IDEAL2023 Paper.

[\[Paper Website\]](https://mytuc.org/pxwz)

> If you have any questions please open an issue or find my email adress on this [website](https://mytuc.org/pxwz).

## How to use this code

Tested with python 3.10.11

1. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install the dependencies from `requirements.txt`:
```bash
pip3 install -r requirements.txt
```

3. To download the TinyImagenet-Dataset, have a look at `utils/download_TinyImagenet.py`. Change the configuration of this script at the beginning of the script.
4. To replicate the results from the thesis, have a look at `notebooks/`-directory. The whole codebase is structured as follows: a jupyter-notebook creates a shell script with the configuration of the experiments. The shell scripts start the exeriments and which create log-files. After that you can visualize the logs in another jupyter-notebook
5. Open a notebook and configure the experiment (Especially the parameters `LOG_DIR` and `DATA_DIR`).
6. Run the first part of the notebook to produce a shell-script.
7. Run the shell script
9. If you want to read the code, `src/main.py` is a good place to start

## Example

Replace YOUR_DATA_DIR with the directory where pytorch should download the datasets. Run the code from the project root. It should take ~5 minutes to train on a normal laptop. You should find the logs in `logs/demo/`. I get an accuracy of ~0.2

```
python3 src/main.py \
    --dataset CIFAR10 \
    --num_classes_per_task 2 \
    --num_tasks 5 \
    --seed 0 \
    --memory_size 64 \
    --num_passes 4 \
    --sampler greedy_sampler \
    --encoder cutr_cifar \
    --encoding_block 3 \
    --compressor quantization \
    --n_states 4 \
    --strategy cifar10_transfer \
    --backbone resnet18_cifar \
    --backbone_block 3 \
    --data_dir YOUR_DATA_DIR \
    --log_dir logs \
    --device cpu \
    --exp_name demo
```

## How to replicate results from the paper

Have a look at the following notebooks in the `notebooks` directory. Adapt the parameters to your needs

| figure | related files                                                                     |
| ------ | --------------------------------------------------------------------------------- |
| 2      | `cifar10_rescon.ipynb`, `cifar100_rescon.ipynb`, `plot_comb.ipynb`                |
| 3      | `setup_wts.ipynb`, `plot_wts.ipynb`                                               |
| 4      | `cifar10_fixed_slots.ipynb`, `cifar100_fixed_slots.ipynb`, `plot_eoc.ipynb`       |
| 5      | `cifar10_fixed_memory.ipynb`, `cifar100_fixed_memory.ipynb`, `plot_fixb.ipynb`    |
| 6      | `cifar10_fixed_memory.ipynb`, `cifar100_fixed_memory.ipynb`, `plot_vs_acae.ipynb` |

## Command Line Options

See `src/opts.py` for some of the command line options. Each of the encoders, compressors and classification heads (backbones) have their own set of options. To see them, have a look at the implementation in `src/encoders`,  `src/compressors` and `src/backbones` respectively.

## Acknowledgment

This code is based on [GDumb](https://github.com/drimpossible/GDumb):

> Prabhu, A, Torr, P, Dokania, PGDumb: A Simple Approach that Questions Our Progress in Continual Learning. In The European Conference on Computer Vision (ECCV) 2020.

