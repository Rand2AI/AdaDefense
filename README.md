# AdaDefense
Official implementation of “Gradients Stand-in for Defending Deep Leakage in Federated Learning"

# Introduction

Federated Learning (FL) has become a cornerstone of privacy protection, shifting the paradigm towards localizing sensitive data while only sending model gradients to a central server. This strategy is designed to reinforce privacy protections and minimize the vulnerabilities inherent in centralized data storage systems. Despite its innovative approach, recent empirical studies have highlighted potential weaknesses in FL, notably regarding the exchange of gradients. In response, this study introduces a novel, efficacious method aimed at safeguarding against gradient leakage, namely, ``AdaDefense". Following the idea that model convergence can be achieved by using different types of optimization methods, we suggest using a local stand-in rather than the actual local gradient for global gradient aggregation on the central server. This proposed approach not only effectively prevents gradient leakage, but also ensures that the overall performance of the model remains largely unaffected. Delving into the theoretical dimensions, we explore how gradients may inadvertently leak private information and present a theoretical framework supporting the efficacy of our proposed method. Extensive empirical tests, supported by popular benchmark experiments, validate that our approach maintains model integrity and is robust against gradient leakage, marking an important step in our pursuit of safe and efficient FL.

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/intro.png"width=1000/></div>

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/ad.png"width=1000/></div>

# Requirements

python==3.6.9

torch==1.4.0

torchvision==0.5.0

numpy==1.18.2

tqdm==4.45.0

...

# Performance

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/t1.png"width=1000/></div>

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/t2.png"width=1000/></div>

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/t4.png"width=1000/></div>

<div align=center><img src="https://github.com/Rand2AI/AdaDefense/blob/main/images/t3.png"width=500/></div>

# Citation

If you find this work helpful for your research, please cite the following paper:
```
@INPROCEEDINGS{adadefense,
    author={Hu, Yi and Ren, Hanchi and Hu, Chen and Li, Yiming and Deng, Jingjing and Xie, Xianghua},
    booktitle={2023 IEEE International Conference on Computing in Natural Sciences, Biomedicine and Engineering},
    title={Gradients Stand-in for Defending Deep Leakage in Federated Learning},
    year={2024},
    volume={},
    number={},
    pages={},
    keywords={Federated learning;Neural networks;Distributed databases;Robustness;Reproducibility of results;Optimization;Convergence;Federated Learning;Weights Aggregation;Adaptive Learning},
    doi={}}
```