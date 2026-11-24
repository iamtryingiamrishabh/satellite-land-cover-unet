# Land Use / Land Cover Segmentation (DeepGlobe)

U-Net (ResNet34 encoder, ImageNet-pretrained) trained to classify every pixel
of high-resolution satellite imagery into 6 land cover classes: urban,
agriculture, rangeland, forest, water, barren land.

## Results (121 held-out test images)

| Metric | Value |
|---|---|
| Pixel accuracy | 0.870 |
| Mean IoU | 0.706 |
| Mean F1 | 0.815 |

| Class | IoU | F1 |
|---|---|---|
| Urban | 0.773 | 0.872 |
| Agriculture | 0.866 | 0.928 |
| Rangeland | 0.380 | 0.551 |
| Forest | 0.781 | 0.877 |
| Water | 0.825 | 0.904 |
| Barren | 0.610 | 0.758 |

![Confusion matrix](figures/confusion_matrix.png)
![Example prediction](figures/EXAMPLE.png)

## Method
- Dataset: DeepGlobe Land Cover (Kaggle, 803 labelled 2448x2448 images).
  Only the train folder has masks, so I made my own 70/15/15 split by image.
- Images cut into 512x512 patches (14,050 train / 3,000 val / 3,025 test).
- Loss: cross-entropy + Dice; the "unknown" class is ignored. Mixed precision, Adam.
- Full tiles are predicted with a sliding window (50% overlap, averaged
  probabilities) to avoid seams.
- Trained on a free Colab T4 GPU.

## Limitations
- Rangeland is the weakest class: it is confused with agriculture (22% of
  rangeland pixels) and forest (9%). The classes look alike in RGB and the
  ground-truth masks are coarse and partly subjective.
- Trained on ~50 cm RGB imagery. It will not transfer directly to Sentinel-2.

## Run it
Open `lulc_deepglobe.ipynb` in Colab with a T4 GPU. Model weights: [https://colab.research.google.com/drive/1_IiCxFRO7WgbLeOljGJfcti_oE0ajqqQ#scrollTo=X4SjDtK5_ttv].
