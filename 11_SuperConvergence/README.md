# SuperConvergence
> Nikhil Shrimali

## Model - <a href = 'https://github.com/nikshrimali/shrinkai/blob/master/model/model11.py'>DavidNet</a>
- Batch size - 512
- One Cycle LR
> ![LR Finder](./assets/lr_finder.png)

- Augmentations - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
> ![Augmentations](./assets/augmentations.png)

- Model's Performance

> ![Model Metrics](./assets/model_metrics.png)