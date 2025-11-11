# Implementation of "Delayed Feedback Modeling with Influence Functions" (AAAI 2026)

📖 Paper: Delayed Feedback Modeling with Influence Functions (AAAI 2026).
🔗 Paper Link: https://arxiv.org/pdf/2502.01669?

✍️ Chenlu Ding, Jiancan Wu, Yancheng Yuan, Cunchun Li, Xiang Wang, Dingxian Wang, Frank Yang, and Andrew Rabinovich
🌸 This code draws on the code of https://github.com/yfwang2021/ULC, including the training of the vanilla model and the implementation of the baselines. Thanks for their code.

## Preparation
1. Prepare the environment:
```bash
git clone https://github.com/oceanoceanna/IF-DFM.git
cd IF-DFM
pip install -r requirements.txt
```

2. Prepare the vanilla model
The trained vanilla model (demo) is in ./seed_train/MLP
You can also train your own vanilla model, incorporating samples with different training durations and varying numbers of false negative samples, which can be constructed via the main.py.

3. Download the data
Download the [Criteo dataset](https://drive.google.com/file/d/1x4KktfZtls9QjNdFYKCjTpfjM4tG2PcK/view).

4. Prepare the data and checkpoints:
Place data.txt in the /data directory.
Ensure that the data path and log path are correct！

## Get start

```cd src```

```python ./cal_delta_para.py --x_lr 0.0001  --x_adjust 0.001  --x_init_value 0.0001 --x_batch 2048 --test_gap 10 --training_duration 14 --training_end_day 15 --cuda_device 1 --valid_test_size 1 --base_model MLP --seed 0```


## Hyperparameters
- $x_lr$: Learning rate of the optimization algorithm.
- $x_adjust$: Regularization term.
- $x_init_value$: Initial value of the parameter change.
- $x_batch$: Batch size.
- $test_gap$: $d_test$ in paper.
- $valid_test_size$: The size of the validation/test sets.
- $base_model$: Choose from [MLP, DeepFM, AutoInt, DCNv2].
