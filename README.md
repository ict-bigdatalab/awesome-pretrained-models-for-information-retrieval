<p align="center">
  <br>
  <img width="300" src="./imgs/logo.svg" alt="logo of awesome repository">
  <br>
  <br>
</p>

# awesome-pretrained-models-for-information-retrieval 

> A curated list of awesome papers related to pre-trained models for information retrieval (a.k.a., **pre-training for IR**). If I missed any papers, feel free to open a PR to include them! And any feedback and contributions are welcome! 



## Pre-training for IR

- [Survey Papers](#survey-papers)
- [Phase 1: First-stage Retrieval](#first-stage-retrieval)
  <details>
  <summary>
  <a href="#sparse-retrieval">Sparse Retrieval </a>

  </summary>

    - [Neural term re-weighting](#neural-term-re-weighting)
    - [Query or document expansion](#query-or-document-expansion)
    - [Sparse representation learning](#sparse-representation-learning)
    <!-- - [Combining neural term re-weighting and document expansion](#combining-neural-term-re-weighting-and-document-expansion) -->
  </details>

  <details>
  <summary>
    <a href="#dense-retrieval">Dense Retrieval </a>
  </summary>

    - [Hard negative sampling](#hard-negative-sampling)
    - [Late interaction and multi-vector representation](#late-interaction-and-multi-vector-representation)
    - [Knowledge distillation](#knowledge-distillation)
    - [Pre-training tailored for dense retrieval](#pre-training-tailored-for-dense-retrieval)
    - [Jointly learning retrieval and indexing](#jointly-learning-retrieval-and-indexing)
    - [Domain adaptation](#domain-adaptation)
    - [Query reformulation](#query-reformulation)
    - [Bias](#bias)
  </details>

  <details>
  <summary>
    <a href="#hybrid-retrieval">Hybrid Retrieval </a>
  </summary>

  </details>


- [Phase 2: Re-ranking Stage](#re-ranking-stage)
  <details>
  <summary>
    <a href="#basic-usage">Basic Usage </a>
  </summary>

    - [Discriminative ranking models](#discriminative-ranking-models)
    - [Generative ranking models](#generative-ranking-models)
    - [Hybrid ranking models](#hybrid-ranking-models)
  </details>

  <details>
  <summary>
    <a href="#long-document-processing-techniques">Long Document Processing Techniques </a>
  </summary>

    - [Passage score aggregation](#passage-score-aggregation)
    - [Passage representation aggregation](#passage-representation-aggregation)
    - [Designing new architectures](#designing-new-architectures)
  </details>

  <details>
  <summary>
      <a href="#improving-efficiency">Improving Efficiency </a>
  </summary>

    - [Decoupling the interaction](#decoupling-the-interaction)
    - [Knowledge distillation](#knowledge-distillation)
    - [Partial Fine-tuning](#partial-fine-tuning)
    - [Early exit](#early-exit)
  </details>

  <details>
  <summary>
      <a href="#other-topics">Other Topics </a>
  </summary>

  - [Query Expansion](#query-expansion)
  - [Re-weighting Training Samples](#re-weighting-training-samples)
  - [Pre-training Tailored for Re-ranking](#pre-training-tailored-for-re-ranking)
  - [Adversarial Attack and Defence](#adversarial-attack-and-defence)
  - [Cross-lingual Retrieval](#cross-lingual-retrieval)
  </details>

- [Jointly Learning Retrieval and Re-ranking](#jointly-learning-retrieval-and-re-ranking)
- [Model-based IR System](#model-based-ir-system)

- [Multimodal Retrieval](#multimodal-retrieval)
  <details>
  <summary>
    <a href="#unified-single-stream-architecture">Unified Single-stream Architecture </a>
  </summary>

  </details>

  <details>
  <summary>
      <a href="#multi-stream-architecture-applied-on-input">Multi-stream Architecture Applied on Input </a>
  </summary>

  </details>

- [Other Resources](#other-resources)



 
## Survey Papers
- [Pre-training Methods in Information Retrieval.](https://arxiv.org/pdf/2111.13853.pdf) *Yixing Fan, Xiaohui Xie et.al.*  FnTIR 2022
- [Dense Text Retrieval based on Pretrained Language Models: A Survey.](https://arxiv.org/pdf/2211.14876.pdf) *Wayne Xin Zhao, Jing Liu et.al.* Arxiv 2022
- [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/abs/2010.06467) *Jimmy Lin et.al.*  M&C 2021
- [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) *Jiafeng Guo et.al.* TOIS 2021
- [A Deep Look into neural ranking models for information retrieval.](https://arxiv.org/abs/1903.06902) *Jiafeng Guo et.al.* IPM 2020




## First Stage Retrieval

### Sparse Retrieval
#### Neural term re-weighting
- [Learning to Reweight Terms with Distributed Representations.](https://dl.acm.org/doi/pdf/10.1145/2766462.2767700) *Guoqing Zheng, Jamie Callan* SIGIR 2015.(**DeepTR**)
- [Context-Aware Term Weighting For First Stage Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204) *Zhuyun Dai et.al.* SIGIR 2020 short. [[code](https://github.com/AdeDZY/DeepCT)] (**DeepCT**)
- [Context-Aware Document Term Weighting for Ad-Hoc Search.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258) *Zhuyun Dai et.al.* WWW 2020. [[code](https://github.com/AdeDZY/DeepCT/tree/master/HDCT)] (**HDCT**)
- [Learning Term Discrimination.](https://arxiv.org/pdf/2004.11759.pdf) *Jibril Frej et.al.* SIGIR 2020. (**IDF-reweighting**)
- [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List.](https://arxiv.org/pdf/2104.07186.pdf) *Luyu Gao et.al.* NAACL 2020. [[code](https://github.com/luyug/COIL)] (**COIL**)
- [Learning Passage Impacts for Inverted Indexes.](https://arxiv.org/pdf/2104.12016.pdf) *Antonio Mallia et.al.* SIGIR 2021 short. [[code](https://github.com/DI4IR/SIGIR2021)] (**DeepImapct**)


#### Query or document expansion
- [Document Expansion by Query Prediction.](https://arxiv.org/pdf/1904.08375.pdf) *Rodrigo Nogueira et.al.* [[doc2query code](https://github.com/nyu-dl/dl4ir-doc2query), [docTTTTTquery code](https://github.com/castorini/docTTTTTquery)] (**doc2query, docTTTTTquery**)
- [Generation-Augmented Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2009.08553.pdf) *Yuning Mao et.al.* ACL 2021. [[code](https://github.com/morningmoni/GAR)] (**query expansion with BART**)


<!-- #### Combining neural term re-weighting and document expansion -->
#### Sparse representation learning
- [SparTerm: Learning Term-based Sparse Representation for Fast Text Retrieval.](https://arxiv.org/pdf/2010.00768.pdf) *Yang Bai, Xiaoguang Li et.al.* Arxiv 2020. (**SparTerm: Term importance distribution from MLM+Binary Term Gating**)
- [Contextualized Sparse Representations for Real-Time Open-Domain Question Answering.](https://arxiv.org/pdf/1911.02896.pdf) *Jinhyuk Lee, Minjoon Seo et.al.* ACL 2020. [[code](https://github.com/jhyuklee/sparc)] (**SPARC, sparse vectors**)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.](https://arxiv.org/pdf/2107.05720.pdf), and [v2.](https://arxiv.org/pdf/2109.10086.pdf) *Thibault Formal et.al.* SIGIR 2021. [[code](https://github.com/naver/splade)](**SPLADE**)
- [Ultra-High Dimensional Sparse Representations with Binarization for Efficient Text Retrieval.](https://arxiv.org/pdf/2104.07198.pdf) *Kyoung-Rok Jang et.al.* EMNLP 2021. (**UHD**)
- [Efficient Passage Retrieval with Hashing for Open-domain Question Answering.](https://arxiv.org/pdf/2106.00882.pdf) *Ikuya Yamada et.al.* ACL 2021. [[code](https://github.com/studio-ousia/bpr)] (**BPR, convert embedding vector to binary codes**)



### Dense Retrieval


#### Hard negative sampling
- [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) *Vladimir Karpukhin,Barlas Oguz et.al.* EMNLP 2020 [[code](https://github.com/facebookresearch/DPR)] (**DPR, in-batch negatives**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et.al.* Arxiv 2020. [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) *Lee Xiong, Chenyan Xiong et.al.* [[code](https://github.com/microsoft/ANCE)] (**ANCE, refresh index during training**)
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et.al.* NAACL 2021. (**RocketQA: cross-batch negatives, denoise hard negatives and data augementation**)
- [Optimizing Dense Retrieval Model Training with Hard Negatives.](https://arxiv.org/pdf/2104.08051.pdf) *Jingtao Zhan et.al.* SIGIR 2021.[[code](https://github.com/jingtaozhan/DRhard)] (**ADORE&STAR, query-side finetuning build on pretrained document encoders**)
- [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) *Sebastian Hofstätter et.al.* SIGIR 2021.[[code](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval)] (**TAS-Balanced, sample from query cluster and distill from BERT ensemble**)
- [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://arxiv.org/pdf/2108.06027.pdf) *Ruiyang Ren et.al.* EMNLP Findings 2021. [[code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)] (**PAIR**)



#### Late interaction and multi-vector representation
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) *Omar Khattab et.al.* SIGIR 2020. [[code](https://github.com/stanford-futuredata/ColBERT)] (**ColBERT**)
- [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) *Samuel Humeau,Kurt Shuster et.al.* ICLR 2020. [[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder)] (**Poly-encoders**)
- [Sparse, Dense, and Attentional Representations for Text Retrieval.](https://arxiv.org/pdf/2005.00181.pdf) *Yi Luan, Jacob Eisenstein et.al.* TACL 2020. (**ME-BERT, multi-vectors**)
- [Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval.](https://arxiv.org/pdf/2105.03599.pdf) *Hongyin Tang, Xingwu Sun et.al.* ACL 2021.
- [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index.](https://arxiv.org/pdf/1906.05807.pdf) *Minjoon Seo,Jinhyuk Lee et.al.* ACL 2019. [[code](https://github.com/uwnlp/denspi)] (**DENSPI**)
- [Learning Dense Representations of Phrases at Scale.](https://arxiv.org/pdf/2012.12624.pdf) *Jinhyuk Lee, Danqi Chen et.al.* ACL 2021. [[code](https://github.com/jhyuklee/DensePhrases)] (**DensePhrases**)
- [Multi-View Document Representation Learning for Open-Domain Dense Retrieval.](https://arxiv.org/pdf/2203.08372.pdf) *Shunyu Zhang et.al.* ACL 2022. (**MVR**)


#### Knowledge distillation
- [Distilling Knowledge from Reader to Retriever for Question Answering.](https://arxiv.org/pdf/2012.04584.pdf) *Gautier Izacard, Edouard Grave.* ICLR 2020. [[unofficial code](https://github.com/lucidrains/distilled-retriever-pytorch)] (**Distill cross-attention of reader to retriever**)
- [Distilling Knowledge for Fast Retrieval-based Chat-bots.](https://arxiv.org/pdf/2004.11045.pdf) *Amir Vakili Tahami et.al.* SIGIR 2020. [[code](https://github.com/KamyarGhajar/DistilledNeuralResponseRanker)] (**Distill from cross-encoders to bi-encoders**)
- [Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation.](https://arxiv.org/pdf/2010.02666.pdf) *Sebastian Hofstätter et.al.* Arxiv 2020. [[code](https://github.com/sebastian-hofstaetter/neural-ranking-kd)] (**Distill from BERT ensemble**)
- [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers.](https://arxiv.org/pdf/2010.11386.pdf) *Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin.* Arxiv 2020. [[code](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert.md)] (**TCTColBERT: distill from ColBERT**)
- [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) *Sebastian Hofstätter et.al.* SIGIR 2021.[[code](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval)] (**TAS-Balanced, sample from query cluster and distill from BERT ensemble**)
- [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking.](https://arxiv.org/pdf/2110.07367.pdf) *Ruiyang Ren, Yingqi Qu et.al.* EMNLP 2021. [[code](https://github.com/PaddlePaddle/RocketQA)] (**RocketQAv2, joint learning by distillation**)
- [Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3477495.3531961) *Kelong Mao et.al.* SIGIR 2022.

 
#### Pre-training tailored for dense retrieval
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et.al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [Pre-training tasks for embedding-based large scale retrieval.](https://arxiv.org/pdf/2002.03932.pdf) *Wei-Cheng Chang et.al.* ICLR 2020. (**ICT, BFS and WLP**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et.al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder.](https://arxiv.org/pdf/2102.09206.pdf) *Shuqi Lu, Di He, Chenyan Xiong et.al.* EMNLP 2021. [[code](https://github.com/microsoft/SEED-Encoder/)] (**Seed**)
- [Condenser: a Pre-training Architecture for Dense Retrieval.](https://arxiv.org/pdf/2104.08253.pdf) *Luyu Gao et.al.* EMNLP 2021. [[code](https://github.com/luyug/Condenser)](**Condenser**)
- [Unsupervised Context Aware Sentence Representation Pretraining for Multi-lingual Dense Retrieval.](https://arxiv.org/pdf/2206.03281.pdf) *Ning Wu et.al.* JICAI 2022. [[code](https://github.com/wuning0929/CCP_IJCAI22)](**CCP, cross-lingual pre-training**)
- [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval.](https://arxiv.org/pdf/2108.05540.pdf) *Luyu Gao et.al.* ACL 2022. [[code](https://github.com/luyug/Condenser)](**coCondenser**)
- [LaPraDoR: Unsupervised Pretrained Dense Retriever for Zero-Shot Text Retrieval.](https://arxiv.org/pdf/2203.06169.pdf) *Canwen Xu, Daya Guo et.al.* ACL 2022. [[code](https://github.com/JetRunner/LaPraDoR)] (**LaPraDoR, ICT+dropout**)
- [A Contrastive Pre-training Approach to Learn Discriminative Autoencoder for Dense Retrieval.](https://arxiv.org/pdf/2208.09846.pdf) *Xinyu Ma et.al.* CIKM 2022. (**CPADE, document term distribution-based contrastive pretraining**)
- [Pre-train a Discriminative Text Encoder for Dense Retrieval via Contrastive Span Prediction](https://arxiv.org/pdf/2204.10641.pdf) *Xinyu Ma et.al.* SIGIR 2022. [[code](https://github.com/Albert-Ma/COSTA)](**COSTA, group-wise contrastive learning**)
- [H-ERNIE: A Multi-Granularity Pre-Trained Language Model for Web Search.](https://dl.acm.org/doi/pdf/10.1145/3477495.3531986) *Xiaokai Chu et.al.* SIGIR 2022. (**H-ERNIE**)
- [Structure and Semantics Preserving Document Representations.](https://arxiv.org/pdf/2201.03720.pdf) *Natraj Raman et.al.* SIGIR 2022.
- [Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning.](https://arxiv.org/pdf/2112.09118.pdf) *Gautier Izacard et.al.* TMLR 2022. [[code](https://github.com/facebookresearch/contriever)] (**Contriever**)


#### Jointly learning retrieval and indexing
- [Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index.](https://arxiv.org/pdf/2105.03933.pdf) *Han Zhang et.al.* SIGIR 2021 short. [[code](https://github.com/jdcomsearch/poeem)] (**Poeem**)
- [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance.](https://arxiv.org/pdf/2108.00644.pdf) *Jingtao Zhan et.al.* CIKM 2021. [[code](https://github.com/jingtaozhan/JPQ)] (**JPQ**)
- [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval.](https://arxiv.org/pdf/2110.05789.pdf)*Jingtao Zhan et.al.* WSDM 2022. [[code](https://github.com/jingtaozhan/RepCONC)] (**RepCONC**)
- [Matchingoriented Embedding Quantization For Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.07858.pdf) *Shitao Xiao et.al.* EMNLP 2021. [[code](https://github.com/microsoft/MoPQ)]
- [Distill-VQ: Learning Retrieval Oriented Vector Quantization By Distilling Knowledge from Dense Embeddings.](https://arxiv.org/pdf/2204.00185.pdf) *Shitao Xiao et.al.* SIGIR 2022. [[code](https://github.com/staoxiao/LibVQ)]


#### Multi-hop dense retrieval
- [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval.](https://arxiv.org/pdf/2009.12756.pdf) *Wenhan Xiong, Xiang Lorraine Li et.al.* ICLR 2021 [[code](https://github.com/facebookresearch/multihop_dense_retrieval)] (**Iteratively encode the question and previously retrieved documents as query vectors**)

#### Domain adaptation
- [Multi-Task Retrieval for Knowledge-Intensive Tasks.](https://arxiv.org/pdf/2101.00117.pdf) *Jean Maillard, Vladimir Karpukhin^ et.al.*  ACL 2021. (**Multi-task learning**)
- [Evaluating Extrapolation Performance of Dense Retrieval.](https://arxiv.org/pdf/2204.11447.pdf) *Jingtao Zhan et.al.* CIKM 2022. [[code](https://github.com/jingtaozhan/extrapolate-eval)]


#### Query reformulation
- [PseudoRelevance Feedback for Multiple Representation Dense Retrieval.](https://arxiv.org/pdf/2106.11251.pdf) *Xiao Wang et.al.* ICTIR 2021 (**ColBERT-PRF**)
- [Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback.](https://arxiv.org/pdf/2108.13454.pdf) *HongChien Yu et.al.* CIKM 2021. [[code](https://github.com/yuhongqian/ANCE-PRF)] (**ANCE-PRF**)
- [LoL: A Comparative Regularization Loss over Query Reformulation Losses for Pseudo-Relevance Feedback.](https://arxiv.org/pdf/2204.11545.pdf) *Yunchang Zhu et.al.* SIGIR 2022. [[code](https://github.com/zycdev/LoL)] (**LoL, Pseudo-relevance feedback**)



#### Bias
- [Implicit Feedback for Dense Passage Retrieval: A Counterfactual Approach.](https://arxiv.org/pdf/2204.00718.pdf) *Shengyao Zhuang et.al.* SIGIR 2022. [[code](https://github.com/ielab/Counterfactual-DR)] (**CoRocchio, Counterfactual Rocchio algorithm**)
- [Hard Negatives or False Negatives: Correcting Pooling Bias in Training Neural Ranking Models.](https://arxiv.org/pdf/2209.05072.pdf) *Yinqiong Cai et.al.* CIKM 2022.



### Hybrid Retrieval
- [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index.](https://arxiv.org/pdf/1906.05807.pdf) *Minjoon Seo,Jinhyuk Lee et.al.* ACL 2019. [[code](https://github.com/uwnlp/denspi)] (**DENSPI**)
- [Complement Lexical Retrieval Model with Semantic Residual Embeddings.](https://arxiv.org/pdf/2004.13969.pdf) *Luyu Gao et.al.* ECIR 2021.
- [BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3471158.3472233) *Shuai Wang et.al.* ICTIR 2021.
- [Progressively Optimized Bi-Granular Document Representation for Scalable Embedding Based Retrieval.](https://arxiv.org/pdf/2201.05409.pdf) *Shitao Xiao et.al.* WWW 2022. [[code](https://github.com/microsoft/BiDR)]



## Re-ranking Stage


### Basic Usage

#### Discriminative ranking models

##### Representation-focused
- [Understanding the Behaviors of BERT in Ranking.](https://arxiv.org/pdf/1904.07531.pdf) *Yifan Qiao et.al.* Aixiv 2019. (**Representation-focused and Interanction-focused**)

##### Interanction-focused
- [Passage Re-ranking with BERT.](https://arxiv.org/pdf/1901.04085.pdf) *Rodrigo Nogueira et.al.* [[code](https://github.com/nyu-dl/dl4marco-bert)] (**monoBERT: Maybe the first work on applying BERT to IR**)
- [Multi-Stage Document Ranking with BERT,](https://arxiv.org/pdf/1910.14424.pdf) [The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models.](https://arxiv.org/pdf/2101.05667.pdf) *Rodrigo Nogueira et.al.* Arxiv 2020. (**Expando-Mono-Duo: doc2query+pointwise+pairwise**)
- [CEDR: Contextualized Embeddings for Document Ranking.](https://arxiv.org/pdf/1904.07094.pdf) *Sean MacAvaney et.al.* SIGIR 2020 short. [[code](https://github.com/Georgetown-IR-Lab/cedr)] (**CEDR: BERT+neuIR model**)


#### Generative ranking models
- [Beyond [CLS] through Ranking by Generation.](https://arxiv.org/pdf/2010.03073.pdf) *Cicero Nogueira dos Santos et.al.* EMNLP 2020 short. (**Query generation using GPT and BART**)
- [Document Ranking with a Pretrained Sequence-to-Sequence Model.](https://arxiv.org/pdf/2003.06713.pdf) *Rodrigo Nogueira, Zhiying Jiang et.al.* EMNLP 2020. [[code](https://github.com/castorini/pygaggle/)] (**Relevance token  generation using T5**)


#### Hybrid ranking models
- [Generalizing Discriminative Retrieval Models using Generative Tasks.](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1414) *Bingsheng Liu, Hamed Zamani et.al.* WWW 2021. (**GDMTL,joint discriminative and generative model with multitask learning**)



### Long Document Processing Techniques
#### Passage score aggregation
- [Deeper Text Understanding for IR with Contextual Neural Language Modeling.](https://arxiv.org/pdf/1905.09217.pdf) *Zhuyun Dai et.al.* SIGIR 2020 short. [[code](https://github.com/AdeDZY/SIGIR19-BERT-IR)] (**BERT-MaxP, BERT-firstP, BERT-sumP: Passage-level**)
- [Simple Applications of BERT for Ad Hoc Document Retrieval,](https://arxiv.org/pdf/1903.10972.pdf) [Applying BERT to Document Retrieval with Birch,](https://www.aclweb.org/anthology/D19-3004.pdf) [Cross-Domain Modeling of Sentence-Level Evidence for Document Retrieval.](https://www.aclweb.org/anthology/D19-1352.pdf) *Wei Yang, Haotian Zhang et.al.* Arxiv 2020, *Zeynep Akkalyoncu Yilmaz et.al.* EMNLP 2019 short. [[code](https://github.com/castorini/birch)] (**Birch: Sentence-level**)
- [Intra-Document Cascading: Learning to Select Passages for Neural Document Ranking.](https://arxiv.org/pdf/2105.09816.pdf) *Sebastian Hofstätter et.al.* SIGIR 2021. [[code](https://github.com/sebastian-hofstaetter/intra-document-cascade)] (**Distill a ranking model to conv-knrm to select top-k passages**)


#### Passage representation aggregation
- [PARADE: Passage Representation Aggregation for Document Reranking.](https://arxiv.org/pdf/2008.09093.pdf) *Canjia Li et.al.* Arxiv 2020. [[code](https://github.com/canjiali/PARADE/)] (**An extensive comparison of various Passage Representation Aggregation methods**)
- [Leveraging Passage-level Cumulative Gain for Document Ranking.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380305) *Zhijing Wu et.al.* WWW 2020. (**PCGM**)


#### Designing new architectures
- [Local Self-Attention over Long Text for Efficient Document Retrieval.](https://arxiv.org/pdf/2005.04908.pdf) *Sebastian Hofstätter et.al.* SIGIR 2020 short. [[code](https://github.com/sebastian-hofstaetter/transformer-kernel-ranking)] (**TKL:Transformer-Kernel for long text**)
- [Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching.](https://arxiv.org/pdf/2004.12297v2.pdf) *Liu Yang et.al.* CIKM 2020. [[code](https://github.com/google-research/google-research/tree/master/smith)] (**SMITH for doc2doc matching**)
- [Socialformer: Social Network Inspired Long Document Modeling for Document Ranking.](https://arxiv.org/pdf/2202.10870.pdf) *Yujia Zhou et.al.* WWW 2022. (**Socialformer**)


### Improving Efficiency

#### Decoupling the interaction
- [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) *Yuyu Zhang, Ping Nie et.al.* SIGIR 2020 short. (**DC-BERT**)
- [Efficient Document Re-Ranking for Transformers by Precomputing Term Representations.](https://arxiv.org/pdf/2004.14255.pdf) *Sean MacAvaney et.al.* SIGIR 2020. [[code](https://github.com/Georgetown-IR-Lab/prettr-neural-ir)] (**PreTTR**)
- [Modularized Transfomer-based Ranking Framework.](https://arxiv.org/pdf/2004.13313.pdf) *Luyu Gao et.al.* EMNLP 2020. [[code](https://github.com/luyug/MORES)] (**MORES, similar to PreTTR**)
- [TILDE: Term Independent Likelihood moDEl for Passage Re-ranking.](https://dl.acm.org/doi/pdf/10.1145/3404835.3462922) *Shengyao Zhuang, Guido Zuccon* SIGIR 2021. [[code](https://github.com/ielab/TILDE)] (**TILDE**)
- [Fast Forward Indexes for Efficient Document Ranking.](https://arxiv.org/pdf/2110.06051.pdf) *Jurek Leonhardt et.al.* WWW 2022. (**Fast forward index**)


#### Knowledge distillation
- [Understanding BERT Rankers Under Distillation.](https://arxiv.org/pdf/2007.11088.pdf) *Luyu Gao et.al.* ICTIR 2020. (**LM Distill + Ranker Distill**)
- [Simplified TinyBERT: Knowledge Distillation for Document Retrieval.](https://arxiv.org/pdf/2009.07531.pdf) *Xuanang Chen et.al.* ECIR 2021. [[code](https://github.com/cxa-unique/Simplified-TinyBERT)] (**TinyBERT+knowledge distillation**)


#### Partial Fine-tuning
- [Semi-Siamese Bi-encoder Neural Ranking Model Using Lightweight Fine-Tuning.](https://arxiv.org/pdf/2110.14943.pdf) *Euna Jung, Jaekeol Choi et.al.* WWW 2022. [[code](https://github.com/xlpczv/Semi_Siamese)] (**Lightweight Fine-Tuning**)
- [Scattered or Connected? An Optimized Parameter-efficient Tuning Approach for Information Retrieval.](https://arxiv.org/pdf/2208.09847.pdf) *Xinyu Ma et.al.* CIKM 2022.(**IAA, introduce the aside module to stabilize training**)


#### Early exit
- [The Cascade Transformer: an Application for Efficient Answer Sentence Selection.](https://arxiv.org/pdf/2005.02534.pdf) *Luca Soldaini et.al.* ACL 2020.[[code](https://github.com/alexa/wqa-cascade-transformers)] (**Cascade Transformer: prune candidates by layer**)
- [Early Exiting BERT for Efficient Document Ranking.](https://www.aclweb.org/anthology/2020.sustainlp-1.11.pdf) *Ji Xin et.al.* EMNLP 2020 SustaiNLP Workshop. [[code](https://github.com/castorini/earlyexiting-monobert)] (**Early exit**)



### Other Topics

#### Query Expansion 
- [BERT-QE: Contextualized Query Expansion for Document Re-ranking.](https://arxiv.org/pdf/2009.07258.pdf) *Zhi Zheng et.al.* EMNLP 2020 Findings. [[code](https://github.com/zh-zheng/BERT-QE)] (**BERT-QE**)


#### Re-weighting Training Samples
- [Training Curricula for Open Domain Answer Re-Ranking.](https://arxiv.org/pdf/2004.14269.pdf) *Sean MacAvaney et.al.* SIGIR 2020. [[code](https://github.com/Georgetown-IR-Lab/curricula-neural-ir)] (**curriculum learning based on BM25**)
- [Not All Relevance Scores are Equal: Efficient Uncertainty and Calibration Modeling for Deep Retrieval Models.](https://arxiv.org/pdf/2105.04651.pdf) *Daniel Cohen et.al.* SIGIR 2021.

#### Pre-training Tailored for Re-ranking

- [MarkedBERT: Integrating Traditional IR Cues in Pre-trained Language Models for Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401194) *Lila Boualili et.al.* SIGIR 2020 short. [[code](https://github.com/BOUALILILila/markers_bert)] (**MarkedBERT**)
- [Selective Weak Supervision for Neural Information Retrieval.](https://arxiv.org/pdf/2001.10382.pdf) *Kaitao Zhang et.al.* WWW 2020. [[code](https://github.com/thunlp/ReInfoSelect)] (**ReInfoSelect**)
- [PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2010.10137.pdf) *Xinyu Ma et.al.* WSDM 2021. [[code](https://github.com/Albert-Ma/PROP)] (**PROP**)
- [Cross-lingual Language Model Pretraining for Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3442381.3449830) *Puxuan Yu et.al.* WWW 2021. 
- [B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.09791.pdf) *Xinyu Ma et.al.* SIGIR 2021. [[code](https://github.com/Albert-Ma/PROP)] (**B-PROP**)
- [Pre-training for Ad-hoc Retrieval: Hyperlink is Also You Need.](https://arxiv.org/pdf/2108.09346.pdf) *Zhengyi Ma et.al.* CIKM 2021. [[code](https://github.com/zhengyima/Anchors)] (**HARP**)
- [Contrastive Learning of User Behavior Sequence for Context-Aware Document Ranking.](https://arxiv.org/pdf/2108.10510.pdf) *Yutao Zhu et.al.* CIKM 2021. [[code](https://github.com/DaoD/COCA)](**COCA**)
- [Pre-trained Language Model based Ranking in Baidu Search.](https://arxiv.org/pdf/2105.11108.pdf) *Lixin Zou et.al.* KDD 2021.
- [A Unified Pretraining Framework for Passage Ranking and Expansion.](https://ojs.aaai.org/index.php/AAAI/article/view/16584) *Ming Yan et.al.* AAAI 2021. (**UED, jointly training ranking and query generation**)
- [Axiomatically Regularized Pre-training for Ad hoc Search.](https://xuanyuan14.github.io/files/SIGIR22Chen.pdf) *Jia Chen et.al.* SIGIR 2022. [[code](https://github.com/xuanyuan14/ARES)] (**ARES**)
- [Webformer: Pre-training with Web Pages for Information Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3477495.3532086) *Yu Guo et.al.* SIGIR 2022. (**Webformer**)



#### Adversarial Attack and Defence
- [Competitive Search.](https://dl.acm.org/doi/pdf/10.1145/3477495.3532771) *Oren Kurland et.al.* SIGIR 2022.
- [PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models.](https://arxiv.org/pdf/2204.01321) *Chen Wu et.al.* Arxiv 2022
- [Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models](https://arxiv.org/pdf/2209.06506.pdf) *Jiawei Liu et.al.* CCS 2022
- [Are Neural Ranking Models Robust?](https://arxiv.org/pdf/2108.05018.pdf) *Chen Wu et.al.* TOIS
- [Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models](https://arxiv.org/pdf/2209.06691.pdf) *Chen Wu et.al.* CIKM 2022




#### Cross-lingual Retrieval
- [Cross-lingual Retrieval for Iterative Self-Supervised Training.](https://arxiv.org/pdf/2006.09526.pdf) *Chau Tran et.al.* NIPS 2020. [[code](https://github.com/pytorch/fairseq/tree/master/examples/criss)] (**CRISS**)
- [CLIRMatrix: A massively large collection of bilingual and multilingual datasets for Cross-Lingual Information Retrieval.](https://www.aclweb.org/anthology/2020.emnlp-main.340.pdf) *Shuo Sun et.al.* EMNLP 2020. [[code](https://github.com/ssun32/CLIRMatrix)] (**Multilingual dataset-CLIRMatrix and multilingual BERT**)




## Jointly Learning Retrieval and Re-ranking
- [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking.](https://arxiv.org/pdf/2110.07367.pdf) *Ruiyang Ren, Yingqi Qu et.al.* EMNLP 2021. [[code](https://github.com/PaddlePaddle/RocketQA)] (**RocketQAv2**)
- [Adversarial Retriever-Ranker for dense text retrieval.](https://arxiv.org/pdf/2110.03611.pdf) *Hang Zhang et.al.* ICLR 2022. [[code](https://github.com/microsoft/AR2)] (**AR2**)
- [RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows.](https://dl.acm.org/doi/pdf/10.1145/3477495.3532050) *Jiarui Qin et.al.* SIGIR 2022. (**RankFlow**)



## Model-based IR System
- [Rethinking Search: Making Domain Experts out of Dilettantes.](https://arxiv.org/pdf/2105.02274.pdf) *Donald Metzler et.al.* SIGIR Forum 2020. 
(**Envisioned the model-based IR system**)
- [Transformer Memory as a Differentiable Search Index.](https://arxiv.org/pdf/2202.06991.pdf) *Yi Tay et.al.* Arxiv 2022. 
(**DSI**)
- [DynamicRetriever: A Pre-training Model-based IR System with Neither Sparse nor Dense Index.](https://arxiv.org/pdf/2203.00537.pdf) *Yujia Zhou et.al.* Arxiv 2022. 
(**DynamicRetriever**)
- [A Neural Corpus Indexer for Document Retrieval.](https://arxiv.org/pdf/2206.02743.pdf) *Yujing Wang et.al.* Arxiv 2022. (**NCI**)
- [Autoregressive Search Engines: Generating Substrings as Document Identifiers.](https://arxiv.org/pdf/2204.10628.pdf) *Michele Bevilacqua et.al.* Arxiv 2022. [[code](https://github.com/facebookresearch/SEAL)] (**SEAL**)
- [CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks.](https://arxiv.org/pdf/2208.07652.pdf) *Jiangui Chen et.al.* CIKM 2022. [[code](https://github.com/ict-bigdatalab/CorpusBrain)] (**CorpusBrain**)


## Multimodal Retrieval


### Unified Single-stream Architecture
- [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training.](https://arxiv.org/pdf/1908.06066.pdf) *Gen Li, Nan Duan et.al.* AAAI 2020.  [[code](https://github.com/microsoft/Unicoder)] (**Unicoder-VL**)
- [XGPT: Cross-modal Generative Pre-Training for Image Captioning.](https://arxiv.org/pdf/2003.01473.pdf) *Qiaolin Xia, Haoyang Huang, Nan Duan et.al.* Arxiv 2020.  [[code](https://github.com/microsoft/Unicoder)] (**XGPT**)
- [UNITER: UNiversal Image-TExt Representation Learning.](https://arxiv.org/pdf/1909.11740.pdf) *Yen-Chun Chen, Linjie Li et.al.* ECCV 2020.  [[code](https://github.com/ChenRocks/UNITER)] (**UNITER**)
- [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks.](https://arxiv.org/pdf/2004.06165.pdf) *Xiujun Li, Xi Yin et.al.* ECCV 2020.  [[code](https://github.com/microsoft/Oscar)] (**Oscar**)
- [VinVL: Making Visual Representations Matter in Vision-Language Models.](https://arxiv.org/pdf/2101.00529.pdf) *Pengchuan Zhang, Xiujun Li et.al.* ECCV 2020.  [[code](https://github.com/microsoft/Oscar)] (**VinVL**)
- [Dynamic Modality Interaction Modeling for Image-Text Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3404835.3462829) *Leigang Qu et.al.* SIGIR 2021 **Best student paper**. [[code](https://sigir21.wixsite.com/dime)] (**DIME**)

### Multi-stream Architecture Applied on Input
- [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.](https://arxiv.org/pdf/1908.02265.pdf) *Jiasen Lu, Dhruv Batra et.al.* NeurIPS 2019.  [[code](https://github.com/facebookresearch/vilbert-multi-task)] (**VilBERT**)
- [12-in-1: Multi-Task Vision and Language Representation Learning.](https://arxiv.org/pdf/1912.02315.pdf) *Jiasen Lu, Dhruv Batra et.al.* CVPR 2020.  [[code](https://github.com/facebookresearch/vilbert-multi-task)] (**A multi-task model based on VilBERT**)
- [Learning Transferable Visual Models From Natural Language Supervision.](https://arxiv.org/pdf/2103.00020.pdf) *Alec Radford et.al.* CVPR 2020.  [[code](https://github.com/OpenAI/CLIP)] (**CLIP, GPT team**)
- [ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph.](https://arxiv.org/pdf/2006.16934.pdf) *Fei Yu, Jiji Tang et.al.* Arxiv 2020. [[code](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-vil)]  (**ERNIE-ViL，1st place on the VCR leaderboard**)
- [M6-v0: Vision-and-Language Interaction for Multi-modal Pretraining.](https://arxiv.org/pdf/2003.13198.pdf) *Junyang Lin, An Yang et.al.* KDD 2020.  (**M6-v0/InterBERT**)
- [M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training.](https://arxiv.org/pdf/2006.02635.pdf) *Haoyang Huang, Lin Su et.al.* CVPR 2021. [[code](https://github.com/microsoft/M3P)]  (**M3P, MILD dataset**)


## Other Resources

### Some Retrieval Toolkits
- [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
- [Pyserini: a Python Toolkit to Support Sparse and Dense Representations](https://github.com/castorini/pyserini/)
- [MatchZoo: a library consisting of many popular neural text matching models](https://github.com/NTMC-Community/MatchZoo)

### Other Resources About Pre-trained Models in NLP
- [Pre-trained Models for Natural Language Processing: A Survey.](https://arxiv.org/abs/2003.08271) *Xipeng Qiu et.al.* 
- [BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers)
- [Pre-trained Languge Model Papers from THU-NLP](https://github.com/thunlp/PLMpapers)

### Surveys About Efficient Transformers
- [Efficient Transformers: A Survey.](https://arxiv.org/pdf/2009.06732.pdf) *Yi Tay, Mostafa Dehghani et.al.* Arxiv 2020. 

