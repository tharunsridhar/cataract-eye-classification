PYTHON ?= python

.PHONY: install train evaluate predict app

install:
	$(PYTHON) -m pip install -e . -r requirements.txt

train:
	$(PYTHON) -m cataract_classifier.training.train --model $(MODEL)

evaluate:
	$(PYTHON) -m cataract_classifier.evaluation.evaluate --model $(MODEL)

predict:
	$(PYTHON) -m cataract_classifier.inference.predict_cli --image $(IMAGE) --strategy weighted

app:
	streamlit run app/streamlit_app.py
