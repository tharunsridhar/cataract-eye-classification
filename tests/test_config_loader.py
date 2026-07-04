from cataract_classifier.config_loader import resolve_env_vars


def test_resolve_env_vars(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "./models")
    cfg = {"path": "${MODEL_DIR}/vgg19.h5", "items": ["${MODEL_DIR}"]}
    assert resolve_env_vars(cfg) == {"path": "./models/vgg19.h5", "items": ["./models"]}
