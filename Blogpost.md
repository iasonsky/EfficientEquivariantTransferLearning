# (Even More) Efficient Equivariant Transfer Learning from Pretrained Models

Mikhail Vlasenko, Ádám Divák, Iason Skylitsis, Milan Miletić, Zoe Tzifa-Kratira

## Introduction
Equivariance in deep learning refers to a model's ability to maintain consistent output changes in response to specific transformations of the input, ensuring that the model's behavior aligns predictably with the symmetries in the data. Many problems are known to be equivariant in nature, thus using a method that inherently has this inductive bias can increase the robustness and generalization capabilities of the models used. Several very large foundation models have been trained recently in multiple modalities, which deliver unprecedented performance in a wide variety of downstream tasks. These models however are not equivariant by their design, which limits their usability in contexts where this would be necessary. Re-training foundation models from scratch using an equivariant architecture is prohibitively expensive for most researchers, which is why several methods were proposed to get provably equivariant output from non-equivariant backbone architectures. We set out to explore the methods *λ-equitune* and *equizero*  proposed by Basu et al, which were shown to deliver good results in a wide variety of downstream tasks. We perform replication studies, suggest code and parameter improvements that deliver significantly better results, and propose a new alternative method that we call *EquiAttention*. Additionally we explore the performance of these methods on new problems and produce visualizations to better understand their working mechanisms.

## Background
The most well-known equivariance in deep learning is the translation equivariance of Convolutional Neural Networks (CNNs) - an object in the upper left corner of an image has the same visual features as the same object in the lower right corner of an image. Convolutions are a particular layer type that exploit this property, by applying the same computation to different parts of their input. This leads to significantly smaller model sizes than comparable fully connected models due to the inherent weight sharing, and faster and more robust training, as data augmentation is not required to teach equivariance to the model.

In more formal terms, equivariance of model $M$ on data $x$ to transformation $g$ means that 
```math
gM(x) = M(g(x))
```

A related property is invariance, when the output of the model stays the same, regardless of the transformation applied to its input.
```math
M(x) = M(g(x))
```

There are however many problems where equivariance to transformations other than translation is desired. In medical image analysis, protein folding, etc. (add examples here) A lack of equivariance in these domains would mean that even if we know that the model works correctly for all the examples in our test set, it may fail at a slightly modified (e.g. rotated) version of the same inputs. 

Equivariance can in theory be learnt by any model by applying adequate data augmentation during training, simply by providing a wide range of transformed versions of the data set, and expecting a similarly transformed output. This however makes training significantly slower for large data sets, and was shown to still not achieve robust equivariance (reference paper by Erik here). This is why specialized architectures like Group Equivariant Convolutional Networks have been proposed that generalize equivariance to a much wider range of discrete transformations, referred to as groups based on their mathematical description, and these have been shown to perform well in many tasks. 

At the same time, very large foundation models have been trained in self-supervised manner on previously unseen data sizes, like (reference a few papers here). These models achieve state of the art performance on a multitude of downstream tasks, sometimes surpassing specialized solutions in a zero-shot manner, without dedicated training on that particular task. (e.g. CLIP, Flamingo) These models are typically not trained in an equivariant manner, which has led to great interest in transfer learning methods that can equip these models with equivariant properties. (Copy references to a few of the prior works here from the original paper.)

## Overview of the original paper

Basu et al. introduced the *equitune* method as a solution to the challenge of leveraging group equivariance in transfer learning. The proposed methodology of Basu et al. is an equivariant finetuning technique that involves performing group averaging over features that have been extracted from pretrained models.
The core idea behind *equitune* (SOURCE from previous paper) is to incorporate group averaging as a mechanism to align the features extracted from pretrained models with the desired group-equivariant properties. By averaging these features, the network can adapt to new tasks while maintaining group equivariance.
Equitune represents a novel approach to enhancing the transfer learning capabilities of neural networks, particularly in scenarios where group equivariance is a crucial factor. It bridges the gap between pretrained models and group-equivariant architectures, enabling more efficient and reliable transfer learning processes.
However, *equitune* is found to perform poorly when it comes to zero-shot tasks. For this reason, Basu et al. (SOURCE current paper) improve upon Kaba et al. (2022) and introduce *equizero*  as the method that achieves comparatively better results for zero-shot and fine-tuning tasks, if used with the appropriate loss functions. Following up on *equizero* , the authors additionally propose *λ-equitune* as the solution to the observation that pretrained models provide better quality features for certain group transformations compared to others. *λ-equitune* learns importance weights directly from the data and uses them to perform a weighted group averaging, thus leading to better performance compared to simple *equitune*, and competing results to *equizero*  used with the appropriate loss functions. 

### *equizero*  (maybe?)
### *equitune* and *λ-equitune*

## Our contributions - Outline

The original paper proposed two new methods and validated it on an exceptionally wide range of tasks in the domain of vision, natural language processing and reinforcement learning. Having results from such a diverse set of tasks and using multiple backbone models is a strong testament to a well-working method. On the other hand we noticed that the publication included a different subset of the transfer learning methods for different tasks, so we wanted to verify whether the results also hold for the missing experiments. The publication included a limited discussion of the weight patterns *λ-equitune* learns, but it was based on a plot created for a single training example, which clearly does not generalize and makes it impossible to draw meaningful conclusions. Additionally we also noticed that many of the tasks chosen, for example image classification, were not equivariant but invariant in their nature, so good results on these does not necessarily verify true equivariance of the solution. These observations motivated us to perform reproducibility studies on some of the original data sets, expand the discussion of the inner workings of *λ-equitune*, and to perform similar studies on additional data that tests the equivariant properties more.

In addition we noticed that even the most sophisticated method proposed, *λ-equitune*, inspects each feature map individually when calculating the weight, disregarding a significant source of information. Given the enormous success of the Transformer architecture in almost all areas of deep learning in recent years (quote at least Attention is all you need + one review), we hypothesized that using an Attention layer instead might provide more flexibility and thus better performance. This motivated us to create an extension of the original methods called *EquiAttention*.

The rest of the blog post will be structured accordingly to provide a summary of our methodologies and results 
( put an outline here based on the actual headers we will have )

### Reproducibility and minor implementation improvements
The authors kindly shared their implementation of the paper’s methods and experiments, which formed the basis of our work. We started by reproducing the experiments related to image classification and we were pleased to find that we could recreate Figure 4 from the original publication easily. Some training parameters were not specified in the publication, in which case we used default values in the code base unless otherwise noted. However, upon closer examination of the implementation, we discovered multiple points that we believe could be improved in the implementation.

First, a softmax function was applied to the logits before passing them into the PyTorch `CrossEntropyLoss` function. Since `CrossEntropyLoss` internally applies a softmax, this additional softmax acts as a smoothing function, hindering the model’s ability to predict sharp distributions and slowing down training, as it decreases the gradients. Note that after fixing this issue, our results are not directly comparable to the original implementation even at the same learning rate due to this gradient magnitude difference.

Training was done in two phases: first only the weight network is trained, while keeping the backbone frozen (even the layers that come after the weight network), which is referred to as pre-finetuning. Then the whole network is fine-tuned, which is simply referred to as fine-tuning. The fine-tuning step is different from what the authors have originally used, as in this step they kept the weight network frozen and only trained the backbone. We found no theoretical justification for this approach, and also found it to perform worse in practice, so we kept the weight network trainable during finetuning. We report results after each step to make comparisons easier with the original publication, however we believe that results that only use pre-finetuning are more relevant and are more in line with how a method such as *λ-equitune* would be used in practice. Especially when trying to achieve equivariance on a special task like medical segmentation, where typically only limited training samples are available, keeping the backbone network frozen and only training the weight network can seriously lower the risk of overfitting.

Our experiments show that removing the redundant softmax and adopting end-to-end finetuning significantly improve performance. With these changes, along with using a lower learning rate of $5e-8$ for better training stability, we achieve results that surpass those reported in the original paper by a large margin. We achieved an increase of 12.5 percentage points (52.6%) in Top1 accuracy on CIFAR100 when using 90 degree rotations as the group transformations and only training the weight network (pre-finetuning), as can be seen in the table below. The increase of 5.95 percentage points (11.8%) is less pronounced but still significant in case of full finetuning, and a the results are equally satisfactory across all other tests we have performed.

|    | Architecture-Transformation   |   CIFAR100 Original Prefinetune Top1 Acc |   CIFAR100 Updated Prefinetune Top1 Acc |   CIFAR100 Original Finetune Top1 Acc |   CIFAR100 Updated Finetune Top1 Acc |   ISIC2018 Original Prefinetune Top1 Acc |   ISIC2018 Updated Prefinetune Top1 Acc |   ISIC2018 Original Finetune Top1 Acc |   ISIC2018 Updated Finetune Top1 Acc |
|---:|:------------------------------|-----------------------------------------:|----------------------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|--------------------------------------:|-------------------------------------:|
|  0 | RN50 flip                     |                                    36.12 |                                   37.94 |                                 52.04 |                                56.49 |                                    12.94 |                                   13.38 |                                 35.75 |                                53.89 |
|  1 | RN50 rot90                    |                                    23.75 |                                   36.25 |                                 50.63 |                                56.58 |                                    12.81 |                                   13.5  |                                 37.31 |                                54.4  |


## Introducing equivariance into the CLIP image classification experiments
Upon a closer inspection of the implementation of the EquiCLIP experiments, we also noticed an important discrepancy between the equations described in the paper and the actual algorithm implemented in the codebase. While the paper described *λ-equitune* by performing a group inverse transformation on the output of each separate backbone model before averaging the feature maps, in practice the code implementation simply took an average of the logits calculated by each backbone without any inverse operation. Please see the equations below for a precise comparison of the mathematics of the paper and the code.

Equations described in the publication
```math
\mathbf{M}_G^\lambda(x) = \frac{1}{\sum_{g \in G} \lambda(gx)} \sum_{g \in G}^{|G|} g^{-1} \lambda(gx) \mathbf{M}(gx).
```

Equations the describe the code (derived by us):
```math
\mathbf{M}_{g\in G}^\lambda(x) = \lambda(\mathbf{M}(gx)) \mathbf{M}(gx) \\
```
```math
\text{class\_sim}_{g\in G}^\lambda = \text{prompt\_embeddings} \cdot \mathbf{M}_{g\in G}^\lambda(x)
```
```math
\text{logits}_{g\in G}^\lambda = softmax(\text{class\_sim}_{g\in G}^\lambda)
```
```math
\text{output}_{G}^\lambda = \frac{1}{|G|} \sum_{g \in G}^{|G|} \text{logits}_{g}^\lambda
```

In a correspondence with the authors they shared that this is because the image classification experiments do not require equivariance, only invariance, and it was not even possible to apply the inverse group transformation to the logits (the final outputs) of the backbone models in this case, as those have no spatial meaning anymore. While we certainly agree with these observations, we were interested in understanding how the truly equivariant method (as described in the paper) would perform, so proceeded to make the necessary changes to the code. It is at this point that we would like to note that *λ-equitune* was not implemented as a single generic framework that could be applied as a post-processing step on any backbone, but was copied and adapted for each experiment individually.

In order to test true equivariance in an image processing setting, we modified the implementation to only run each backbone until the last convolutional layer, performed the inverse transformation, weight calculation and feature combination on this spatial features, then passed the resulting combined feature map through the remaining layers of the backbone network without changing them. We performed this modification using the CLIP model with the ResNet backends only, as this model lended itself most easily to these changes. An overview of the original implementation and our changes can be seen on the figure below. It is important to note that the feature maps for the group-transformed input images are not just transformed versions of the feature map of the original image, so applying the inverse transformation does not yield 4 identical feature maps.

![Architecture diagrams of a non-equivariant network, *λ-equitune* using the original implementation and our version of it](images/architecture_diagrams.svg)

By applying these changes and testing with 90 degree rotations as the group transformation, we achieved an increase of xx in Top1 accuracy on CIFAR100 when using 90 degree rotations as the group transformations and only training the weight network (pre-finetuning), as can be seen in the table below. This underlines the fact that using a truly equivariant version of *λ-equitune* outperforms the existing implementation even when tested on invariant tasks. 
In (section x) we explore the performance of this method on truly equivariant tasks.

### EquiAttention: Using Attention as a feature combination method
In the original work, the weights of features from $gx$ in the average are obtained independently for each $gx$. We see this as a potential limitation, as such an approach is withholding potentially crucial information for determining the significance of specific features. 

We note that the only requirement for obtaining equivariant weights in the given setting is maintaining equivariance for permutation of feature sets. Specifically,

```math
\boldsymbol\pi(f([\forall g \in G: \mathbf{M_1}(gx))]) = f(\boldsymbol\pi([\forall g \in G: \mathbf{M_1}(gx))]))
```

for an array permutation operator $\boldsymbol\pi$ and a function $f$ that produces an array weights from an array of features.

One permutation equivariant transformation that is being successfully applied across modalities is Attention. For this reason, we attempt to improve results further by utilizing an attention-based framework for computing the weights of feature sets.

Attention is calculated using the usual formulation of queries $Q$, keys $K$ and values $V$ (insert reference here):
```math
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

for feature sets $H = [h_0, h_1, \dots, h_{|G|-1}]$ ($h$ because these are hiddens) and arbitrary index $i \in [0, |G|-1]$. $h_i$ is a feature set obtained by $\mathbf{M_1}(g_ix)$

```math
Q_i = QNet(h_i)
```

```math
K_i = KNet(h_i)
```

```math
V_i = g_i^{-1}h_i
```

where values $V$ are the inputs with the inverse transformation applied to them.

Using the Attention as described above, we can calculate the final output of EquiAttention as follows:
```math
\mathbf{M}_G^A(x) = \mathbf{M}_2(\frac{1}{|G|}\sum_{g \in G}^{|G|} \text{Attention\_module}([\mathbf{M_1}(g_0x), \dots, \mathbf{M_1}(g_{|G|-1}x)]))
```

where `Attention_module` takes the features sets and applies one attention operation as described above.

Using the above described method of EquiAttention, we achieved a result of .. (insert results here)

### Visualizations: understanding what *λ-equitune* (and *EquiAttention*) learns
Add visualizations here

## Replicability: verifying the effectiveness on new problems
Replicating the results on novel datasets which exhibit different properties is an important step in verifying the effectiveness of any new method. This is why, instead of reproducing all results from the original publication, we decided to perform replication on 2 (3?) new data sets: the ISIC2018 image classification and the (describe extended NLG task). (and describe object detection as well if we include it)
ISIC 2018 Medical Imaging dataset
Image classification using widely used benchmarks like ImageNet and CIFAR provides a helpful understanding of the performance of the methodologies, as it places the result within the context of the multitude of other methods that have been tested on the same datasets. However these images have a natural orientation, so making models trained on them equivariant is less important. This is why we chose to test on a medical imaging dataset, where a natural orientation does not exist and any rotations of the inputs are equally likely, and equivariance of the detections is important. Thus, we use the ISIC 2018 dataset, which was published by the International Skin Imaging Collaboration (ISIC) as a large-scale dataset of dermoscopy images.The dataset consists from 10015 training images and 194 validation images belonging to seven distinct diagnostic categories: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis / Bowen’s disease (intraepithelial carcinoma), Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), Dermatofibroma, and Vascular lesion. An example of the dataset images is shown below: 

![Sample of classes in ISIC 2018 data set](images/isic2018_sample.png)

Image classification was performed by finetuning CLIP with a Resnet 50 backbone. It can be seen in the results that [whichever method works better - add results].

### Extended NLG task

Additionally, the authors formalized a group-theoretic approach to fairness in Natural Language Generation (NLG) task. Previous work has shown that Large Language Models (LLMs), such as GPT-2, are biased towards certain demographic groups in their generations. While there was notable effort put into evaluating bias in LLMs (Sheng et al, 2019), little has been done to theoretically study the mitigation of this bias.

Basu et al (2023) introduced a novel approach to fairness in NLG using group theory (full details in the Appendix). The model is given a context $X1$ (e.g. *'The man was known for'*) and asked to generate a continuation $X2$. They define a cyclic group $G_2 = {e, g}$ \footnote{this is a simplification for the binary setting} with a group action g that swaps the demographic identifier in a given text. Thus, $gX1$ will become *'The woman was known for'*. They call a model group-theoretically fair if:

```math
\forall g \in G: P(gX2 | gX1) = P(X2 | X1)
```

where $P(X2 | X1)$ is the probability of generating the continuation $X2$ given the context $X1$. Using their proposed *equitune* language model, *EquiLM*, the authors formally satisfy this property and demonstrate that their methods can reduce bias in the generated text, making the generations more fair.

However, as mentioned above, the authors only focused on establishing fairness across binary demographic groups. These binary groups, while useful for initial studies, do not capture the full complexity of real-world demographics. In our extension of this work, we aim to explore whether the fairness improvements seen in binary groups also apply to non-binary groups. All of the three considered demographic groups naturally extend beyond binary classifications. <Add here explicitly how we extend the three demographic groups once we decide>. By extending the fairness framework to non-binary groups, we can better reflect the diversity of human identities and ensure that the proposed methods can mitigate bias in real-world settings.

<Add results once available> 


## Summary
Equivariant fine-tuning of large foundational models is an attractive approach for combining the unprecedented representation capabilities of these models with the introduction of guaranteed equivariance required in certain domains. We have reproduced, replicated and extended the work of Basu et al., where they introduced *equizero*  and *λ-equitune* for performing such fine-tuning. We have achieved an increase of [xx pp] in top1 accuracy on [whichever dataset] by improving code and parameters, a further increase of [xx pp] by improving the methodology, and proposed a new method called *EquiAttention*, which performed on par with the best baseline. Additionally, we have verified the efficacy of these methods on novel datasets that exhibit equivariant properties and delivered visualizations to better understand the operation of the trained *λ-equitune* and *EquiAttention* methods. Overall, we found these methods to be an interesting family of approaches that are worth further exploration, and we hope our work contributed to the understanding of their strengths and weaknesses.
Acknowledgements
We would like to thank the authors for making their code available and for their fast and detailed responses to our inquiries. We would also like to thank Yongtuo Liu for his supervision of our work.

## Individual contributions

## References


*** When using the ISIC 2018 datasets in your research, please cite the following works:

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
