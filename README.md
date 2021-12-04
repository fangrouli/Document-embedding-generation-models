# Document-embedding-generation-models

  
## **Introduction**

This is a repository for a project: Development and Application of Document Embedding for Semantic Text Retrieval

We utilised multiple sentence embedding generation techniques and the SBERT sentence encoder (https://www.sbert.net/) to generate document embeddings based on hierarchical doucment structure.

The models that we developed:
1. Baseline (Average Pool)
2. CNN (based on TextCNN by Kim, 2014)
3. Transformer (based on Transformer model by Vaswani *et al.*, 2017)
4. Transformer-Poly-Encoder
5. Poly-Encoder (based on Poly-encoders by Humeau *et al.*, 2019)

The models are evaluated and trained on PatentMatch ultrabalanced datasets (by Risch *et al.*, 2020).

The training instance is a pair of documents (or paragraphs), _text_ and _text_b_. If the two is related (e.g. _text_ cited _text_b_), the label is 1, otherwise 0.

The documents will be broken into sentences, which will be tokenized and encoded by SBERT. The set of sentence embeddings will then be the input of the models, which will turn them into document embeddings (i.e. paragraph embeddings) for similarity score calculation.

The similarity score calculation we used is cosine similarity, and model evaluation metric is AUROC.

  
## **Step To Run Source Codes**
1. Run DataPrep.py, enter the file directories of the original .tsv files of the train and test ultrabalanced PatentMatch dataset.
2. Run TokGen.py, clearing of data and convert them to index tokens.
3. Run ValidationSet.py, generate validation dataset from the training dataset.
4. Run Baseline.py, evaluate the dataset and create the score.pt to keep logging model performance.
5. Run any model from CNN.py, Transformer.py, Trans-poly-encoder.py and Poly-encoder.py according to need. (Note that Trans-poly-encoder.py requires a pretrained transformer model).
  
### **Extra Files**
parameters.py: Configuration files, as well as some universally used functions.  
ModelScore.py: The function for AUC score generation for model evaluation. 

  
## **Reference**
1. SBERT Sentence Encoder (https://www.sbert.net/): Reimers & Gurevych, 2019, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, arXiv e-prints, URL: https://arxiv.org/abs/1908.10084 ;
2. TextCNN: Kim, 2014, *Convolutional Neural Networks for Sentence Classification*, arXiv e-prints, URL:https://arxiv.org/abs/1408.5882 ;
3. Transformer: Vaswani *et al.*, 2017, *Attention is All You Need*, URL: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf ;
4. Poly-Encoder: Humeau *et al.*, 2019, *Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring*, arXiv e-prints, URL: https://arxiv.org/abs/1905.01969 ;
5. PatentMatch Dataset: Risch *et al.*, 2020, *PatentMatch: A Dataset for Matching Patent Claims & Prior Art*, arXiv e-prints, URL: https://arxiv.org/abs/2012.13919 ;
6. References for the tranformer and poly-encoder model construction are cited in the corresponding python files as comments.


