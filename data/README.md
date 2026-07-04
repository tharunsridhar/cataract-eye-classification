# Dataset Layout

The code expects `DATA_DIR` to point at a folder with this structure:

```text
DATA_DIR/
  train/
    Immature/
    Mature/
    Normal/
  valid/
    Immature/
    Mature/
    Normal/
  test/
    Immature/
    Mature/
    Normal/
```

Do not commit bulk medical image data to this repository. Keep datasets local,
or link to the original licensed dataset source.
