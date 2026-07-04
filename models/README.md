# Model Weights

Place trained model files here, or track them with Git LFS.
The original accuracy-tagged files were normalized to stable registry names.

Expected files:

| File | Architecture | Reported Test Accuracy |
|---|---|---:|
| `densenet121.h5` | DenseNet121 | 99.03 |
| `efficientnetb3.h5` | EfficientNetB3 | 99.84 |
| `resnet50.h5` | ResNet50 | 99.91 |
| `vgg19.h5` | VGG19 | 99.89 |

The active paths are configured in `configs/model_registry.yaml`.
