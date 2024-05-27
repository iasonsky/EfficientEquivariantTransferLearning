## Original vs Updated Full Val Summary for CIFAR100

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                  31.42 |               51.17 |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                  35.12 |               56.15 |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                  37.07 |               54.04 |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                  37.69 |               55.64 |

*Table 1: Image classification results using the author's original and our modified code base*