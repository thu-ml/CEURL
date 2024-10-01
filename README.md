# CEURL

[![arXiv](https://img.shields.io/badge/arXiv-2405.14073-b31b1b.svg)](https://arxiv.org/abs/2405.14073)

This is the Official implementation for "PEAC: Unsupervised Pre-training for Cross-Embodiment Reinforcement Learning" (NeurIPS 2024)

## state-based DMC

The code is based on [URLB](https://github.com/rll-research/url_benchmark)

```sh
conda create -n ceurl python=3.8
conda activate ceurl
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

```python3
cd DMC_state
chmod +x train_finetune.sh

./train_finetune.sh peac walker_mass 0
./train_finetune.sh peac quadruped_mass 0
./train_finetune.sh peac quadruped_damping 0
```

## Citation

If you find this work helpful, please cite our paper.

```
@article{ying2024peac,
  title={PEAC: Unsupervised Pre-training for Cross-Embodiment Reinforcement Learning},
  author={Ying, Chengyang and Hao, Zhongkai and Zhou, Xinning and Xu, Xuezhou and Su, Hang and Zhang, Xingxing and Zhu, Jun},
  journal={arXiv preprint arXiv:2405.14073},
  year={2024}
}
```