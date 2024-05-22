## Original vs Updated Full Val Summary for ISIC2018

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                35.7513 |             39.8964 |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                15.0259 |             68.3938 |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                45.0777 |             37.8238 |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                17.6166 |             67.8756 |

*Table 2: Image classification results using the author's original and our modified code base*