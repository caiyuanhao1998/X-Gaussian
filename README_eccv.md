## 1. Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# cloning some supported repo from official 3DGS
git clone https://gitlab.inria.fr/sibr/sibr_core.git
cd submodules
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install diff-gaussian-rasterization
cd ..
conda env create --file environment.yml
conda activate x_gaussian
```

## 2. Data Preparation

Download the raw CT data from the [scientific visualization dataset](https://klacansky.com/open-scivis-datasets/). Then install the [TIGRE](https://github.com/CERN/TIGRE) box to process the downloaded data into pickle file. 

Subsequently, put the generated data into the folder `data` and organize them as follows:

```shell
  |--data
      |--chest.pickle
      |--abdomen.pickle
      |--foot.pickle
      |--head.pickle
      |--pancreas.pickle
```

## 3. Training and Evaluation

```shell
bash train.sh
```