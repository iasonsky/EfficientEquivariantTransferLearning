## CIFAR100 Full Validation Accuracy

|    | Architecture-Transformation   |   Original Prefinetune Top1 Acc |   Updated Prefinetune Top1 Acc |   Original Val Top1 Acc |   Updated Val Top1 Acc |   Original Finetune Top1 Acc |   Updated Finetune Top1 Acc |   Original Final Top1 Acc |   Updated Final Top1 Acc |
|---:|:------------------------------|--------------------------------:|-------------------------------:|------------------------:|-----------------------:|-----------------------------:|----------------------------:|--------------------------:|-------------------------:|
|  0 | RN50 rot90                    |                          26.078 |                         35.198 |                   31.63 |                  34.76 |                        52.78 |                       53.85 |                     52.67 |                    53.87 |

**Note:** prefinetune_top1_acc corresponds to the last value it had before finetuning (so it's not the best lambda weights), val_top1_acc has the best lambda weights