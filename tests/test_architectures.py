import pytest

tf = pytest.importorskip("tensorflow")

from cataract_classifier.models.architectures import build_model


@pytest.mark.parametrize("arch", ["vgg19", "resnet50", "densenet121", "efficientnetb3"])
def test_build_model_output_shape(monkeypatch, arch):
    class FakeBase:
        def __init__(self):
            self.input = tf.keras.Input(shape=(224, 224, 3))
            self.output = tf.keras.layers.GlobalAveragePooling2D()(self.input)
            self.layers = []
            self.trainable = False

    monkeypatch.setattr("cataract_classifier.models.architectures.VGG19", lambda **kwargs: FakeBase())
    monkeypatch.setattr("cataract_classifier.models.architectures.ResNet50", lambda **kwargs: FakeBase())
    monkeypatch.setattr("cataract_classifier.models.architectures.DenseNet121", lambda **kwargs: FakeBase())
    monkeypatch.setattr("cataract_classifier.models.architectures.EfficientNetB3", lambda **kwargs: FakeBase())
    model = build_model(arch, num_classes=3, img_size=(224, 224))
    assert model.output_shape[-1] == 3
