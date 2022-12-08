# Notebook Bench

[![Source Code Check](https://github.com/icoxfog417/datascience-template/actions/workflows/ci.yml/badge.svg)](https://github.com/icoxfog417/datascience-template/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black)](https://github.com/PyCQA/flake8)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue)](https://github.com/python/mypy)

Notebooks to compare the performance of platform.

| Benchmark | Task         | Colaboratory | Studio Lab | Kaggle Notebook |
|:----------|:-------------|:-------------|:-----------|:----------------|
| NLP       | Fine tune Transformer model ([cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)) by [Amazon Review dataset](https://huggingface.co/datasets/amazon_reviews_multi) to classify the sentiment. Training epoch is 3. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icoxfog417/notebook-bench/blob/main/notebooks/results/nlp-colaboratory-standard.ipynb) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/icoxfog417/notebook-bench/blob/main/notebooks/nlp.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/icoxfog417/notebook-bench/blob/main/notebooks/results/nlp-kaggle-T4x2.ipynb) |

## Prerequisite

* If the service has already installed PyTorch, we use it.
* If PyTorch is not installed, we install the as same as version.
