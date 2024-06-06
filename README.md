# Handwritten recognition

## Objective

* To **research** possibilities on **handwritten recognition** using **MMOCR 1.0.1**

## Contents

* **[Installation](MMOCR_installation.ipynb)** - installation notebook
* **[Model test](MMOCR_test_model.ipynb)** - model test notebook (incomplete).
* **[Recognition training](MMOCR_training_rec.ipynb)** - text recognition training.
* **[Detection training](MMOCR_training_det.ipynb)** - text detection training (incomplete).
* **[COCO to line dict converter](MMOCR_coco_to_line_dict.ipynb)** - utility for text detection training (mmdet).

## MMOCR description

MMOCR is an utility library connecting mmdet (text detection) and many text recognition libraries (includint SATRN).

Text detection and text recognition trainings are done separately (as much as I know).

For all model adjustements python configs are required. Example custom [config](config/config.py).

## Author

**Kamil Butryn**
