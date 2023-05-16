## Model

Our model is depicted below.

To compare our model with all baselines, we split our workflow into two components: the `embedding` and `classifier`. 

### Embedding
We provide implementations for our model (`meta`) and three baselines (`avg`,`idf`,`cnn`).
- `meta`: depicted above
- `avg`: each example is the average of its word embeddings
- `idf`: each example is the weighted average of its word embeddings, with weights given by each word's idf computed over the source pool
- `cnn`: each example is embedded by a sentence CNN ([Kim 2014](https://www.aclweb.org/anthology/D14-1181 "Kim 2014"))

### Classifier
We provide implementations for
- `proto`: transform the embedding using a MLP and make predictions based on the prototype of each class ([Snell et. al 2017](https://arxiv.org/pdf/1703.05175.pdf "Snell et. al 2017")).
- `mlp`: predict the label using a MLP. When combined with the flag `--maml`, this trains MAML ([Finn et. al 2017](https://arxiv.org/pdf/1703.03400.pdf "Finn et. al 2017")).

