<p align="center">
  <br>
  <img width="200" src="./imgs/logo.svg" alt="logo of awesome repository">
  <br>
  <br>
</p>

# awesome-pretrained-models-for-information-retrieval 

> A curated list of awesome papers related to pre-trained models for information retrieval. Any feedback and contribution are welcome!




## Table of Contents

- [Survey paper](#survey-paper)
- [Phase 1: First-stage retrieval](#first-stage-retrieval)
- [Phase 2: Re-ranking stage](#re-ranking-stage)

For people who want to acquire some basic&advanced knowledge about neural models for information retrieval and try some neural models by hand, we refer readers to the below awesome NeuIR survey and the text-matching toolkit [MatchZoo-py](https://github.com/NTMC-Community/MatchZoo-py):
- [A Deep Look into neural ranking models for information retrieval.](https://arxiv.org/abs/1903.06902) *Guo Jiafeng et.al.* Information Processing & Management, 2020.

 
## Survey Paper
### Pre-trained models
- [Pre-trained Models for Natural Language Processing: A Survey.](https://arxiv.org/abs/2003.08271) *Qiu Xipeng et.al.* 


### Pre-trained models for information retrieval 
- [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/abs/2010.06467) *Jimmy Lin et.al.*


## First Stage Retrieval
- [Traditional ad-hoc retrieval](#traditional-ad-hoc-retrieval)
- [Passage retrieval in open domain question answering](#passage-retrieval-in-open-domain-question-answering)

### Traditional ad-hoc retrieval
- [Context-Aware Term Weighting For First Stage Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204) *Dai Zhuyun et.al.* SIGIR 2020 short. [[code](https://github.com/AdeDZY/DeepCT)] (**DeepCT**)
- [Context-Aware Document Term Weighting for Ad-Hoc Search](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258) *Dai Zhuyun et.al.* WWW 2020. [[code](https://github.com/AdeDZY/DeepCT/tree/master/HDCT)] (**HDCT**)
- [Document Expansion by Query Prediction.](https://arxiv.org/pdf/1904.08375.pdf) *Rodrigo Nogueira et.al.* [[doc2query code](https://github.com/nyu-dl/dl4ir-doc2query),[docTTTTTquery code](https://github.com/castorini/docTTTTTquery)] (**doc2query, docTTTTTquery**)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) *Omar Khattab et.al.* SIGIR 2020. [[code](https://github.com/stanford-futuredata/ColBERT)] (**ColBERT**)
- [Efficient Document Re-Ranking for Transformers by Precomputing Term Representations.](https://arxiv.org/pdf/2004.14255.pdf) *Sean MacAvaney et.al.* SIGIR 2020. [[code](https://github.com/Georgetown-IR-Lab/prettr-neural-ir)] (**PreTTR**)
- [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) *Samuel Humeau,Kurt Shuster et.al.* ICLR 2020. [[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder)] (**Poly-encoders**)
- [Modularized Transfomer-based Ranking Framework](https://arxiv.org/pdf/2004.13313.pdf) *Gao Luyu et.al.* EMNLP 2020. [[code](https://github.com/luyug/MORES)] (**MORES, similar to Poly-encoders**)
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) *Lee Xiong, Chenyan Xiong et.al.* [[code](https://github.com/microsoft/ANCE)] (**ANCE**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et.al.* [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)

### Passage retrieval in open domain question answering
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et.al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et.al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master