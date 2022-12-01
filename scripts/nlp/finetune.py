from typing import Dict

import numpy as np
from scripts.nlp.dataset_preprocessor.amazon_review import AmazonReview
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

# Define pretrained tokenizer and model
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Read data
# About slice https://huggingface.co/docs/datasets/splits.html
review = AmazonReview(
    input_column="review_title",
    label_column="stars",
    tokenizer=tokenizer,
    lang="ja")

dataset = review.load("validation")

dataset = dataset.train_test_split(test_size=0.2)
dataset_train = review.format(dataset["train"])
dataset_validation = review.format(dataset["test"])

print(f"Train data statistics: {review.statistics(dataset_train)}")
print(f"Test data statistics: {review.statistics(dataset_validation)}")

# Define Trainer parameters
def compute_metrics(eval: EvalPrediction) -> Dict[str, float]:
    pred, labels = eval
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define Trainer
args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    seed=0,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()
