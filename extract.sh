#!/bin/bash

python extract_features_m.py \
	--input_file="./data/xa200" \
	--output_file="./data/xa200.jsonl" \
	--vocab_file="../uncased_L-12_H-768_A-12/vocab.txt" \
	--bert_config_file="../uncased_L-12_H-768_A-12/bert_config.json" \
	--init_checkpoint="../uncased_L-12_H-768_A-12/bert_model.ckpt" \
	--layers=-1,-2 \
	--max_seq_length=128 \
	--batch_size=8
