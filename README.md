# NLP-Transformer_XL
An implementation of the [Transformer-XL](https://arxiv.org/abs/1901.02860) in Tensorflow 2.0. A minor difference between this implementation and that in the paper is that the gradient is allowed to propagate through the different segments.

Please note that this repository is still work-in-progress.

## Training
To process the data, first run
```
python process_reddit_jokes_subword.py
```
followed by
```
python train_reddit_jokes_subword_tf_ver2_gpt_xl.py
```
to train the model. Run
```
python infer_reddit_jokes_subword_tf_ver2_gpt_xl.py
```
to perform inference.
