# Real Image Denoising

## Training

* Train WDENet for real image denoising, run
```
python train.py
```

## Evaluation
* Download the pre-trained [models](https://drive.google.com/drive/folders/1bCZVaUpDgA3m2uqiCpMm2F8e1ie2EuDn?usp=share_link) and place them in `./logs/`

* To obtain denoised predictions using WDENet, run
```
python test.py
```