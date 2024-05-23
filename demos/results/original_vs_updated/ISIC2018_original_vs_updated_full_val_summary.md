## Original vs Updated Full Val Summary for ISIC2018

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                  35.75 |               39.9  |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                  15.03 |               68.39 |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                  45.08 |               37.82 |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                  17.62 |               67.88 |

*Table 2: Image classification results using the author's original and our modified code base*