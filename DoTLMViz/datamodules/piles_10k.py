import datasets
import einops
import os
import torch

from typing import Dict, List

from DoTLMViz import DataModule

import DoTLMViz.metadata.piles_10k as metadata


class Piles10k(DataModule):
    """A data module for Piles-10k dataset."""

    def __init__(self, batch_size=256, max_length=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
        self.batch_size = batch_size

        self.data_dir = metadata.DL_DATA_DIRNAME

    def prepare_data(self):
        if not os.path.exists(metadata.PROCESSED_DATA_DIRNAME / f"piles-10k-{self.max_length}"):
            from transformers import GPT2Tokenizer

            dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

            tokenized_dataset = tokenize(dataset, tokenizer, max_length=self.max_length)

            tokenized_dataset.save_to_disk(
                (metadata.PROCESSED_DATA_DIRNAME / f"piles-10k-{self.max_length}").as_posix()
            )

    def setup(self):
        dataset = datasets.load_from_disk((metadata.PROCESSED_DATA_DIRNAME / f"piles-10k-{self.max_length}").as_posix())
        dataset_dict = dataset.train_test_split(test_size=1000)

        self.train_dataset = dataset_dict["train"]["tokens"]
        self.test_dataset = dataset_dict["test"]["tokens"]

    def __repr__(self):
        basic = "Piles-10k Dataset\n"

        return basic


def tokenize(
    dataset,
    tokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 4,
):
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        text = examples[column_name]
        text = tokenizer.eos_token.join(text)

        num_chunks = 20
        chunk_length = len(text) // num_chunks + 1
        chunks = [text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]

        tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"]
        tokens = tokens[tokens != tokenizer.pad_token_id]

        num_batches = len(tokens) // seq_len

        tokens = tokens[: num_batches * seq_len]

        tokens = einops.rearrange(tokens, "(batch seq_len) -> batch seq_len", batch=num_batches)

        if add_bos_token:
            prefix = torch.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = torch.concatenate([prefix, tokens], axis=1)

        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True, num_proc=(num_proc if not streaming else None), remove_columns=[column_name]
    )

    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset
