# Cataract Eye Classification

![Python](https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-F57C00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

Transfer-learning ensemble for cataract severity classification: Immature,
Mature, and Normal.

[Demo video](https://drive.google.com/file/d/1W0x3R28jE3Ym05aut_JKAKl5AUAaLus9/view?usp=sharing)

## Problem

The project classifies eye images into cataract severity classes using trained
CNN backbones and an ensemble prediction layer.

## Architecture

The system uses DenseNet121, EfficientNetB3, ResNet50, and VGG19. See
`docs/ARCHITECTURE.md` for the data flow and ensemble details.

![Pipeline](docs/images/pipeline_diagram.png)

## Results

| Model | Reported Test Accuracy |
|---|---:|
| DenseNet121 | 98.03 |
| EfficientNetB3 | 97.84 |
| ResNet50 | 96.91 |
| VGG19 | 97.89 |

## Repository Structure

```text
configs/                  YAML configs for data, training, models, logging
src/cataract_classifier/  Installable Python package
app/                      Streamlit app
models/                   Local or LFS-tracked model weights
data/                     Local dataset placeholders
docs/                     Architecture and references
assets/                   Sample images and demo media
```

## Setup

```bash
python -m pip install -e . -r requirements.txt
```

Copy `.env.example` to `.env` or export the variables in your shell:

```bash
DATA_DIR=./data/raw
MODEL_DIR=./models
```

## Usage

Train:

```bash
python -m cataract_classifier.training.train --model densenet121
```

Evaluate:

```bash
python -m cataract_classifier.evaluation.evaluate --model vgg19
```

Predict:

```bash
python -m cataract_classifier.inference.predict_cli --image assets/sample_images/sample_normal.jpg --strategy weighted
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

## Dataset

The expected folder layout is documented in `data/README.md`. Bulk medical
image data should stay outside git unless licensing explicitly allows sharing.

## References

See `docs/REFERENCES.md`.

