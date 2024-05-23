## Original vs Updated Full Val Summary for ImagenetV2

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                  46.55 |               46.65 |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                  47.95 |               48.8  |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                  48.75 |               48.4  |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                  53.15 |               53.85 |

*Table 3: Image classification results using the author's original and our modified code base*