# nlp_project

From https://github.com/yala/text_nn based on Tao Lei's Rationalizing neural prediction.

## Requirments
This repository assumes glove embeddings.
Download Glove embeddings at:  https://nlp.stanford.edu/projects/glove/
And place `glove.6B/glove.6B.300d.txt` in `data/embeddings/glove.6B/glove.6B.300d.txt`.

## Usage: 
To train the model with CNN-encoder and RNN-generator :

```
python -u scripts/main.py  --batch_size 64 --cuda --dataset news_group --embedding glove --dropout 0.05 --weight_decay 5e-06 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.0001 --num_workers 0 --objective cross_entropy --patience 50 --save_dir snapshot --train --train --results_path logs/demo_run.results  --gumbel_decay 1e-5 --get_rationales --selection_lambda .001 --continuity_lambda 0 --gen_bidirectional --model_form_gen rnn
```

To train the model with CNN-encoder and CNN-generator:

```
python -u scripts/main.py  --batch_size 64 --cuda --dataset news_group --embedding glove --dropout 0.05 --weight_decay 5e-06 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.0001 --num_workers 0 --objective cross_entropy --patience 50 --save_dir snapshot --train --train --results_path logs/demo_run.results  --gumbel_decay 1e-5 --get_rationales --selection_lambda .001 --continuity_lambda 0 --gen_bidirectional --model_form_gen cnn
```

Words embedding code is in BERT_projet_nlp_rationale.ipynb.
