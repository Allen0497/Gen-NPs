### Training

```
python main.py --cmab_mode=train --model=np --expid=sgld_10000000 --train_mode=s --temperature=10000000
```
If training for the first time, wheel data will be generated and saved in `datasets`. Model weights and logs will be saved in `results/train-all-R`.

### Evaluate
After training, we can run contextual bandit to evaluate the trained model.
```
python main.py --cmab_mode=eval --model=np --expid=sgld_10000000
```
Model weights according to `{expid}` will be loaded and evaluated. If running contextual bandit for the first time, evaluation data wil be generated and saved in `evalsets`. The results will be saved in `results/eval-all-R`.