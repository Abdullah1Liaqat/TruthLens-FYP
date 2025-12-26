import pandas as pd
from transformers import BertTokenizer

LABEL_MAPPING = {
    'pants-fire': 0,
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'mostly-true': 1,
    'true': 1
}

COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
    'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire',
    'context', 'justification'
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LENGTH = 128


def load_liar_dataset(path):
    train_df = pd.read_csv(f"{path}/train.tsv", sep="\t", header=None)
    valid_df = pd.read_csv(f"{path}/valid.tsv", sep="\t", header=None)
    test_df  = pd.read_csv(f"{path}/test.tsv",  sep="\t", header=None)

    for df in [train_df, valid_df, test_df]:
        df.columns = COLUMNS
        df["label_num"] = df["label"].map(LABEL_MAPPING)

    return train_df, valid_df, test_df


def tokenize_texts(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
