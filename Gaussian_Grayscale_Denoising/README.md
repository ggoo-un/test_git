# Gaussian Grayscale Image Denoising

## Training

* Train WDENet for gaussian grayscale image denoising, run
```
python train.py --noiseL 15
python train.py --noiseL 25
python train.py --noiseL 50
```

## Evaluation
* Download the pre-trained [models](https://drive.google.com/drive/folders/1LpIvOZVWYnJFrbSozbduILSYzFgtzl27?usp=share_link) and place them in `./logs/`

* To obtain denoised predictions using WDENet, run
```
python test.py --test_noiseL 15
python test.py --test_noiseL 25
python test.py --test_noiseL 50
```