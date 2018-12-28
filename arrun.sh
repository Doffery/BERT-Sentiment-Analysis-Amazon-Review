#!/bin/bash
export BERT_BASE_DIR=../uncased_L-12_H-768_A-12
python run_classifier_n.py \
  --task_name=AR \
  --do_train=true \
  --do_eval=true \
  --data_dir=./data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=24 \
  --learning_rate=2e-5 \
  --lr_decay=exp \
  --num_train_epochs=7.0 \
  --output_dir=./tmp/tune_small/exp_5e_96_15
#  --use_record
