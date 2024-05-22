## ImagenetV2 Full Validation Accuracy

|    | Architecture-Transformation   |   Original Prefinetune Top1 Acc |   Updated Prefinetune Top1 Acc |   Original Val Top1 Acc |   Updated Val Top1 Acc |   Original Finetune Top1 Acc |   Updated Finetune Top1 Acc |   Original Final Top1 Acc |   Updated Final Top1 Acc |
|---:|:------------------------------|--------------------------------:|-------------------------------:|------------------------:|-----------------------:|-----------------------------:|----------------------------:|--------------------------:|-------------------------:|
|  0 | RN50 rot90                    |                          46.375 |                         49.925 |                   46.55 |                  47.95 |                         46.6 |                        48.2 |                     46.65 |                     48.8 |

**Note:** prefinetune_top1_acc corresponds to the last value it had before finetuning (so it's not the best lambda weights), val_top1_acc has the best lambda weights