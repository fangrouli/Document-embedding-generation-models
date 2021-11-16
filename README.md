# Document-embedding-generation-models

This is a repository for a project: Development and Application of Document Embedding for Semantic Text Retrieval

We utilised multiple sentence embedding generation techniques and the SBERT sentence encoder (https://www.sbert.net/) to generate document embeddings based on hierarchical doucment structure.

The models that we developed:
1. Baseline (Average Pool)
2. CNN (based on TextCNN by Kim, 2014, URL:https://arxiv.org/abs/1408.5882)
3. Transformer (based on Transformer model by Vaswani et al., 2017, URL: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
4. Transformer-Poly-Encoder
5. Poly-Encoder (based on Poly-encoders by Humeau et al., 2019, URL: https://arxiv.org/abs/1905.01969)

The models are evaluated and trained on PatentMatch ultrabalanced datasets (https://arxiv.org/abs/2012.13919).

The training instance is a pair of documents (or paragraphs), _text_ and _text_b_. If the two is related (e.g. _text_ cited _text_b_), the label is 1, otherwise 0.

The documents will be broken into sentences, which will be tokenized and encoded by SBERT. The set of sentence embeddings will then be the input of the models, which will turn them into document embeddings (i.e. paragraph embeddings) for similarity score calculation.

The similarity score calculation we used is cosine similarity, and model evaluation metric is AUROC.
