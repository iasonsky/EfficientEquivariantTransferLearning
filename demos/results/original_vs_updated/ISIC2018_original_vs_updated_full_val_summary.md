## Original vs Updated Full Val Summary for ISIC2018

|    | Method               | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:---------------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code        | CLIP w RN50 - rot90 - *λ-equitune* |                  15.03 |               63.73 |
|  1 | Updated Code         | CLIP w RN50 - rot90 - *λ-equitune* |                  16.58 |               64.77 |
|  2 | Equivariant Equitune | CLIP w RN50 - rot90 - *λ-equitune* |                  16.58 |               18.13 |

*Table 1: Image classification results using the author's original and our modified code base*