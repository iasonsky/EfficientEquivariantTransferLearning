## Original vs Updated Full Val Summary for CIFAR100

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                  31.63 |               52.67 |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                  34.76 |               53.87 |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                  37.7  |               54.25 |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                  37.7  |               55.09 |

*Table 1: Image classification results using the author's original and our modified code base*