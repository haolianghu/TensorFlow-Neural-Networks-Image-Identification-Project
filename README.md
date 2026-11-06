# MNIST digit classification

`Image Identification.ipynb` demonstrates dense and convolutional neural
networks for classifying 28×28 grayscale handwritten digits. MNIST is
downloaded by Keras at runtime; no image files or Excel workbook are stored in
this repository.

## Reproducible CNN

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mnist_classifier.py --epochs 5 --seed 42 --model mnist_cnn.keras
python -m unittest discover -s tests -v
```

The CLI downloads MNIST on its first run, uses a stratified validation subset,
sets TensorFlow and NumPy seeds, and stops early if validation loss stalls. It
reports accuracy, macro-F1, top-2 accuracy and a 10×10 confusion matrix on the
standard test set. Saving the model is optional; the `.keras` artifact is
ignored by git. Training requires TensorFlow and may take several minutes
depending on available hardware.

The notebook's original dense and CNN models remain exploratory and are not
replaced by the CLI. This familiar benchmark does not establish performance on
other handwriting, devices, or populations. Review per-class metrics and
potential dataset bias before applying a digit recognizer elsewhere.
