# PLANRL: A Motion Planning and Imitation Learning Framework to Bootstrap Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://arxiv.org/abs/2408.04054)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://raaslab.org/projects/NAVINACT/index.html)

## Setup

### Clone the repo
Use `--recursive` to get the correct submodule
```shell
git clone --recursive https://github.com/raaslab/HYBRID-RL.git
```

### Installing libraries
1. Install MuJoCo

    Dowload the MuJoCo version 2.1 for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

    Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`

    ```shell
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    mkdir -p ~/.mujoco
    tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
    rm mujoco210-linux-x86_64.tar.gz
    ```

2. Create conda env

    ```shell
    conda create --name planrl python=3.9
    ```

    Then, source `set_env.sh` to activate `planrl` conda env and setup paths such as `MUJOCO_PY_MUJOCO_PATH` and add current project folder to `PYTHONPATH`
    
    If conda env has a different name, manually modify the env name in `set_env.sh` file. Same case if the mujoco is not installed at default location

    ```shell
    source set_env.sh
    ```

3. Install Python dependencies

    ```shell
    # install pytorch with cuda version 12.1
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    # install dependencies from requirement.txt
    pip install -r requirements.txt
    ```

4. Compile C++ code
    ```
    cd common_utils
    make
    ```

5. Install ompl library - [_will be updated shortly_]


### Trouble Shooting
Later when running the training commands, if we encounter the following error
```shell
ImportError: .../libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
Then we can force the conda to use the system c++ lib.
Use these command to symlink the system c++ lib into conda env. To find `PATH_TO_CONDA_ENV`, run `echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}`.

```shell
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so.6
```


## Reproduce Results


Download dataset and models from [Googledrive](www.google.com)

Put the folders under `release` folder. The release folder shoudl contain `release/cfgs`(already in the repo), `release/data` and `release/model` (from the the dowloaded zip file)

### Metaworld

Train PLANRL
```shell
# assembly
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/planrl.yaml --bc_policy assembly

# boxclose
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/planrl.yaml --bc_policy boxclose

# coffeepush
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/planrl.yaml --bc_policy coffeepush
```

Train IBRL
```shell
# assembly
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl.yaml --bc_policy assembly

# boxclose
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl.yaml --bc_policy boxclose

# coffeepush
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl.yaml --bc_policy coffeepush
```


Train RLMN - RL with ModeNet and NavNet
```shell
# assembly
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/only_rl.yaml --bc_policy assembly

# boxclose
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/only_rl.yaml --bc_policy boxclose

# coffeepush
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/only_rl.yaml --bc_policy coffeepush
```

Run IL with ModeNet and NavNet
```shell
# assembly
python mw_main/mw_planrl_IL.py --config_path release/cfgs/metaworld/only_IL.yaml --bc_policy assembly

# boxclose
python mw_main/mw_planrl_IL.py --config_path release/cfgs/metaworld/only_IL.yaml --bc_policy boxclose

# coffeepush
python mw_main/mw_planrl_IL.py --config_path release/cfgs/metaworld/only_IL.yaml --bc_policy coffeepush
```


Train BC policy
```shell
# assembly
python mw_main/train_bc_mw.py --dataset.path Assembly --save_dir SAVE_DIR
```

<!-- 
Train ModeNet
```shell

```

Train NavNet
```shell

``` -->


Citation
```
@article{bhaskar2024PLANRL,
    title={PLANRL: A Motion Planning and Imitation Learning Framework to Bootstrap Reinforcement Learning},
    author={Bhaskar, Amisha and Mahammad, Zahiruddin and Jadhav, Sachin R and Tokekar, Pratap},
    journal={arXiv preprint arXiv:2408.04054},
    year={2024}
}
```
