## 1D Regression

---
### Training
```
python gp.py --mode=train --expid=default --model=cnp
```

### Evaluation
```
python gp.py --mode=evaluate_all_metrics --expid=default --model=cnp
```

## CelebA Image Completion
---

### Prepare data
Download [img_align_celeba.zip] and unzip. Download [list_eval_partitions.txt] and [identity_CelebA.txt]. Place downloaded files in `datasets/celeba` folder. Run `python data/celeba.py` to preprocess the data.

### Training
```
python celeba.py --mode=train --expid=default-tnpa --model=tnpa
```

### Evaluation
```
python celeba.py --mode=evaluate_all_metrics --expid=default-tnpa --model=tnpa
```

## EMNIST Image Completion
---

### Prepare data
Please modify the `url` field in the `EMNIST` function within the `mnist.py` file of the `torchvision` library to `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip` so that the program can correctly download the EMNIST dataset."

### Training
```
python emnist.py --mode=train --expid=default --model=cnp
```

### Evaluation
```
python emnist.py --mode=evaluate_all_metrics --expid=default --model=cnp
```