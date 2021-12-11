# Document-embedding-generation-models

  
## **Introduction**

This is a repository for a project: Development and Application of Document Embedding for Semantic Text Retrieval

We utilised multiple sentence embedding generation techniques and the SBERT sentence encoder (https://www.sbert.net/) to generate document embeddings based on hierarchical doucment structure.

The pipeline of this project:
  
<img width='500' height='400' src="https://user-images.githubusercontent.com/80878559/145664030-62a4eabe-e910-4852-b5e6-3ddb7602f4a8.png" alt="1">

==============================================================================  
The models that we developed:
1. Baseline (Average Pool)
2. CNN (based on TextCNN by Kim, 2014)
  
Data pipeline of CNN model:  
  
<img width='900' height='300' src="https://user-images.githubusercontent.com/80878559/145664051-b5a62b3b-4f68-432c-b737-00d9f1530a3a.png" alt="2">
  
Image for TextCNN architecture, from Kim, 2014:  
<img width='650' height='300' alt="image" src="https://user-images.githubusercontent.com/80878559/145663994-dcb0f730-79a4-4dd8-8eb8-511b05bedd0a.png">  
    
3. Transformer (based on Transformer model by Vaswani *et al.*, 2017)
  
Data pipeline of the transformer model:  
  
<img width='550' height='300' src="https://user-images.githubusercontent.com/80878559/145664070-39672d1f-65a6-44f4-a2fa-da52b7384828.png" alt="3">
   
4. Transformer-Poly-Encoder  
  
Data pipeline of the Transformer-Poly-Encoder model:  
  
<img width='500' height='300' src="https://user-images.githubusercontent.com/80878559/145664084-4cebfd61-c833-402a-ba8d-4c9194cd1a14.png" alt="4">

5. Poly-Encoder (based on Poly-encoders by Humeau *et al.*, 2019)

Data pipeline of the Poly-Encoder model:  
    
<img width='700' height='300' src="https://user-images.githubusercontent.com/80878559/145664111-933d8f3b-5be2-4684-a56f-c0c13c2df846.png" alt="4">
  
==============================================================================  
The models are evaluated and trained on PatentMatch ultrabalanced datasets (by Risch *et al.*, 2020).

The training instance is a pair of documents (or paragraphs), _text_ and _text_b_. If the two is related (e.g. _text_ cited _text_b_), the label is 1, otherwise 0.

The documents will be broken into sentences, which will be tokenized and encoded by SBERT. The set of sentence embeddings will then be the input of the models, which will turn them into document embeddings (i.e. paragraph embeddings) for similarity score calculation.

The similarity score calculation we used is cosine similarity, and model evaluation metric is AUROC.

  
## **Step To Run Source Codes**
1. Install all the required packages using the requirements.txt .  
2. Run DataPrep.py, enter the file directories of the original .tsv files of the train and test ultrabalanced PatentMatch dataset.
3. Run TokGen.py, clearing of data and convert them to index tokens.
4. Run ValidationSet.py, generate validation dataset from the training dataset.
5. Run Baseline.py, evaluate the dataset and create the score.pt to keep logging model performance.
6. Run any model from CNN.py, Transformer.py, Trans-poly-encoder.py and Poly-encoder.py according to need. (Note that Trans-poly-encoder.py requires a pretrained transformer model).
  
### **Extra Files**
parameters.py: Configuration files, as well as some universally used functions.  
ModelScore.py: The function for AUC score generation for model evaluation.   
DataGenerator.py: The class for data generator used to generate mini-batches and the customized collate function.

  
## **Reference**
1. SBERT Sentence Encoder (https://www.sbert.net/): Reimers & Gurevych, 2019, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, arXiv e-prints, URL: https://arxiv.org/abs/1908.10084 ;
2. TextCNN: Kim, 2014, *Convolutional Neural Networks for Sentence Classification*, arXiv e-prints, URL:https://arxiv.org/abs/1408.5882 ;
3. Transformer: Vaswani *et al.*, 2017, *Attention is All You Need*, URL: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf ;
4. Poly-Encoder: Humeau *et al.*, 2019, *Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring*, arXiv e-prints, URL: https://arxiv.org/abs/1905.01969 ;
5. PatentMatch Dataset: Risch *et al.*, 2020, *PatentMatch: A Dataset for Matching Patent Claims & Prior Art*, arXiv e-prints, URL: https://arxiv.org/abs/2012.13919 ;
6. References for the tranformer and poly-encoder model construction are cited in the corresponding python files as comments.


