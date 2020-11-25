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
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) *Lee Xiong, Chenyan Xiong et.al.* [[code](https://github.com/microsoft/ANCE)] (**ANCE**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et.al.* [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)

### Passage retrieval in open domain question answering
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et.al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://kentonl.com/pub/gltpc.2020.pdf) *Kelvin Guu, Kenton Lee et.al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Pre-training tasks for embedding-based large scale retrieva.](https://arxiv.org/pdf/2002.03932.pdf) *Wei-Cheng Chang et.al.* ICLR 2020. (**ICT, BFS and WLP**)
- [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) *Vladimir Karpukhin,Barlas Oguz et.al.* EMNLP 2020 [[code](https://github.com/facebookresearch/DPR)] (**DPR**)
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et.al.*  (**RocketQA**)
- [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) *Zhang Yuyu, Nie Ping et.al.* SIGIR 2020 short. (**DC-BERT**)

## Re-ranking Stage
- [Directly adapting pre-trained models to IR task](#directly-adapting-pre-trained-models-to-IR-task)
- [Designing new pre-training task](#designing-new-pre-training-task)
- [Designing new pre-training model architecture](#designing-new-pre-training-model-architecture)

### Directly adapting pre-trained models to IR task
- [Passage Re-ranking with BERT.](https://arxiv.org/pdf/1901.04085.pdf) *Rodrigo Nogueira et.al.* [[code](https://github.com/nyu-dl/dl4marco-bert)] (**monoBERT: Maybe the first work on applying BERT to IR**) 
- [Multi-Stage Document Ranking with BERT.](https://arxiv.org/pdf/1910.14424.pdf) *Rodrigo Nogueira et.al.* (**duoBERT: pointwise+pairwise**)
- [Simple Applications of BERT for Ad Hoc Document Retrieval.](https://arxiv.org/pdf/1903.10972.pdf) / [Applying BERT to Document Retrieval with Birch.](https://www.aclweb.org/anthology/D19-3004.pdf) *Wei Yang, Haotian Zhang et.al. / Zeynep Akkalyoncu Yilmaz et.al.* EMNLP 2019 short. [[code](https://github.com/castorini/birch)] (**Birch: Sentence-level**)
- [Deeper Text Understanding for IR with Contextual Neural Language Modeling.](https://arxiv.org/pdf/1905.09217.pdf) *Dai Zhuyun et.al.* SIGIR 2020 short. [[code](https://github.com/AdeDZY/SIGIR19-BERT-IR)] (**BERT-MaxP, BERT-firstP, BERT-sumP: Passage-level**)
- [CEDR: Contextualized Embeddings for Document Ranking.](https://arxiv.org/pdf/1904.07094.pdf) *Sean MacAvaney et.al.* SIGIR 2020 short. [[code](https://github.com/Georgetown-IR-Lab/cedr)] (**CEDR: BERT+ranking model**)
- [Training Curricula for Open Domain Answer Re-Ranking.](https://arxiv.org/pdf/2004.14269.pdf) *Sean MacAvaney et.al.* SIGIR 2020. [[code](https://github.com/Georgetown-IR-Lab/curricula-neural-ir)] (**curriculum learning based on BM25**)
- [Leveraging Passage-level Cumulative Gain for Document Ranking.](http://www.thuir.cn/group/~YQLiu/publications/WWW2020Wu.pdf) *Wu Zhijing et.al.* WWW 2020. (**PCGM**)
- [Selective Weak Supervision for Neural Information Retrieval.](https://arxiv.org/pdf/2001.10382.pdf) *Zhang Kaitao et.al.* WWW 2020. [[code](https://github.com/thunlp/ReInfoSelect)] (**ReInfoSelect**)
- [Document Ranking with a Pretrained Sequence-to-Sequence Model.]() *Rodrigo Nogueira, Zhiying Jiang et.al.* EMNLP 2020. [[code](https://github.com/castorini/pygaggle/)] (**using T5**)
- [Beyond [CLS] through Ranking by Generation.](https://arxiv.org/pdf/2010.03073.pdf) *Cicero Nogueira dos Santos et.al.* EMNLP 2020 short. (**query likelihood computed by GPT**)
- [BERT-QE: Contextualized Query Expansion for Document Re-ranking.](https://arxiv.org/pdf/2009.07258.pdf) *Zhi Zheng et.al.* EMNLP 2020 Findings. [[code](https://github.com/zh-zheng/BERT-QE)] (**BERT-QE**)
- [Cross-lingual Retrieval for Iterative Self-Supervised Training.](https://arxiv.org/pdf/2006.09526.pdf) *Chau Tran et.al.* NIPS 2020. [[code](https://github.com/pytorch/fairseq/tree/master/examples/criss)] (**CRISS**)


### Designing new pre-training task
- [PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2010.10137.pdf) *Ma Xinyu et.al.* WSDM 2021. [[code](https://github.com/Albert-Ma/PROP)] (**PROP**)

### Designing new pre-training model architecture
- [Local Self-Attention over Long Text for Efficient Document Retrieval.](https://arxiv.org/pdf/2005.04908.pdf) *Sebastian Hofstätter et.al.* SIGIR 2020 short. [[code](https://github.com/sebastian-hofstaetter/transformer-kernel-ranking)] (**TKL:Transformer-Kernel for long text**)
- [The Cascade Transformer: an Application for Efficient Answer Sentence Selection.](https://arxiv.org/pdf/2005.02534.pdf) *Luca Soldaini et.al.* ACL 2020.[[code](https://github.com/alexa/wqa-cascade-transformers)] (**Cascade Transformer**)

