# (Even More) Efficient Equivariant Transfer Learning from Pretrained Models

Mikhail Vlasenko, Ádám Divák, Iason Skylitsis, Milan Miletić, Zoe Tzifa-Kratira

## 1. Introduction

Equivariance in deep learning refers to a model's ability to maintain consistent output changes in response to specific
transformations of the input, ensuring that the model's behavior aligns predictably with the symmetries in the data.
Many problems are known to be equivariant in nature, thus using a method that inherently has this inductive bias can
increase the robustness and generalization capabilities of the models used. Several very large foundation models have
been trained recently in multiple modalities, which deliver unprecedented performance in a wide variety of downstream
tasks. These models however are not equivariant by their design, which limits their usability in contexts where this
would be necessary. Re-training foundation models from scratch using an equivariant architecture is prohibitively
expensive for most researchers, which is why several methods were proposed to get provably equivariant output from
non-equivariant backbone architectures. We set out to explore the methods *λ-equitune* and *equizero*  proposed by Basu
et al, which were shown to deliver good results in a wide variety of downstream tasks. We perform replication studies,
suggest code and parameter improvements that deliver significantly better results, and propose a new alternative method
that we call *equiattention*. Additionally we explore the performance of these methods on new problems and produce
visualizations to better understand their working mechanisms.

## 2. Background

The most well-known equivariance in deep learning is the translation equivariance of Convolutional Neural Networks (
CNNs) (LeCun et al. (1989)) - an object in the upper left corner of an image has the same visual features as the same object in the lower
right corner of an image. Convolutions are a particular layer type that exploit this property, by applying the same
computation to different parts of their input. This leads to significantly smaller model sizes than comparable fully
connected models due to the inherent weight sharing, and faster and more robust training, as data augmentation is not
required to teach equivariance to the model.

In more formal terms, equivariance of model $M$ on data $x$ to transformation $g$ means that

```math
gM(x) = M(g(x))
```

A related property is invariance, when the output of the model stays the same, regardless of the transformation applied
to its input.

```math
M(x) = M(g(x))
```

There are however many problems where equivariance to transformations other than translation is desired. Some examples include medical
image segmentation (Bekkers et al. (2018), Lafarge et al. (2021)), protein folding (Tunyasuvunakool et al., 2021), molecule modelling (Hoogeboom et al. (2022)) or modelling a wide range of physical phenomenon (Villar et al. (2021)). A lack of equivariance in these domains would mean that even
if we know that the model works correctly for all the examples in our test set, it may fail at a slightly modified (for example
rotated) version of the same inputs.

Equivariance can in theory be learnt by any model by applying adequate data augmentation during training, simply by
providing a wide range of transformed versions of the data set, and expecting a similarly transformed output. This
however makes training significantly slower for large data sets, and was shown to still not achieve robust
equivariance (Moskalev et al. (2023)). This is why specialized architectures like Group Equivariant Convolutional
Networks (Cohen et al. (2016)) have been proposed that generalize equivariance to a much wider range of discrete transformations, referred to
as groups based on their mathematical description, and these have been shown to perform well in many tasks.

At the same time, very large foundation models have been trained in self-supervised manner on previously unseen data
sizes, like [CLIP](https://openai.com/index/clip/) (Radford et al. (2021)), [GPT-4](https://chatgpt.com/?oai-dm=1) (OpenAI (2023)) or [Llama](https://llama.meta.com/) (Touvron et al. (2023)). These models achieve state of the art performance on a multitude of
downstream tasks, sometimes surpassing specialized solutions in a zero-shot manner, without dedicated training on that
particular task (Bommasani et al. (2021)). These models are typically not trained in an equivariant manner, which has led to
an interest in transfer learning methods that can equip these models with equivariant properties.

## 3. Overview of the original paper

The lack of equivariance of a pretrained model means that upon presenting slightly perturbed versions of the same input,
the output of the model can be widely different. This is especially true for inputs that have a natural orientation or
which for any other reason occur more frequently in a particular configuration in the training set. The main idea behind
the family of methods described in the paper is to create the group-transformed version of the inputs for all
transformations that we are interested in (for example 90-degree rotated versions of the input image), pass each of
these through the same backbone network, and combine the resulting outputs in some way in order to achieve equivariance.
The difference between these methods is in how the final combination step is performed.

<!--
Basu et al. introduced the *equitune* method as a solution to the challenge of leveraging group equivariance in transfer learning. The proposed methodology of Basu et al. is an equivariant finetuning technique that involves performing group averaging over features that have been extracted from pretrained models.
The core idea behind *equitune* (SOURCE from previous paper) is to incorporate group averaging as a mechanism to align the features extracted from pretrained models with the desired group-equivariant properties. By averaging these features, the network can adapt to new tasks while maintaining group equivariance.
Equitune represents a novel approach to enhancing the transfer learning capabilities of neural networks, particularly in scenarios where group equivariance is a crucial factor. It bridges the gap between pretrained models and group-equivariant architectures, enabling more efficient and reliable transfer learning processes.
However, *equitune* is found to perform poorly when it comes to zero-shot tasks. For this reason, Basu et al. (SOURCE current paper) improve upon Kaba et al. (2022) and introduce *equizero*  as the method that achieves comparatively better results for zero-shot and fine-tuning tasks, if used with the appropriate loss functions. Following up on *equizero* , the authors additionally propose *λ-equitune* as the solution to the observation that pretrained models provide better quality features for certain group transformations compared to others. *λ-equitune* learns importance weights directly from the data and uses them to perform a weighted group averaging, thus leading to better performance compared to simple *equitune*, and competing results to *equizero*  used with the appropriate loss functions. 
-->

The *equitune* method that Basu et al. (2023) proposed can turn a non-equivariant model M into a model M_G that is
equivariant under the group actions belonging to the group G, via minimizing the distance of features obtained from
pretrained and equivariant models. The output of an equituned model is given by the following formula:

```math
\mathbf{M}_G(x) = \frac{1}{|G|} \sum_{g \in G} g^{-1} \mathbf{M}(gx).
```

Essentially, this means that the features calculated for each transformed input are averaged with equal weights to
create the final output. Simply averaging the features can lead to detrimental performance, especially in zero-shot
learning tasks, potentially because the pretrained model outputs high quality features only for some of the transformed
inputs.

The *equizero* method introduced by Kaba et al (2022) is formulated as an optimization problem, where all
group-transformed versions of the input are passed through the backbone, but only a single one is selected for producing
the output. More formally:

```math
\mathbf{M}_G(x) = g_{*}^{-1} \mathbf{M}(g_{*}x)
```

where

```math
g_{*} = argmin_{g \in G} l(\mathbf{M}(gx))
```

```math
l : \mathcal{Y} \to \mathbb{R}
```

$l$ is an injective proxy loss function. The choice of $l$ plays an important role in the final zero-shot and finetuning
performance, and one of the contributions of the original publication is showing $l$ functions that work well for
particular problems..

*λ-equitune* is a more general formulation which contains both previous methods as special cases. The main idea of
*λ-equitune* is that given a pretrained model $M$, the features $M(gx)$ for any fixed $x$ are not all equally important
for all $g \in G$. *λ-equitune* learns to assign variable weights to each group-transformed inputs, resulting in the
following formulation:

```math
\mathbf{M}_G^\lambda(x) = \frac{1}{\sum_{g \in G} \lambda(gx)} \sum_{g \in G}^{|G|} g^{-1} \lambda(gx) \mathbf{M}(gx).
```

We get *equitune* as a special case when all λ values are equal, and *equizero* as a special case when λ is an indicator
function. Naturally, *λ-equitune* is implemented as a neural network that learns the $\lambda$ weights,
which can be done with or without fine-tuning the backbone at the same time. As we can see, all methods have a computation cost that grows linearly in the number of group
transformations used.

### 3.2 Related work

The methods described in the original paper fall under the category of *symmetrization*. This means that all transformations of the input are passed through the backbone network and the final output is calculated as some combination of these. A competing approach is *canonicalization*: where a canonicalization network first learns to transform the data into a canonical form, and only this selected form is passed through the network. An architecture based on this idea is described by Mondal et al. (2023). *Canonicalization* has the advantage in that it only requires a single forward pass through the backbone network, so it only has a small computational overhead. On the other hand, the *canonicalization* network has to operate without any input from the backbone network, which may lead to duplicating some low-level image understanding operations and making suboptimal choices, as canonicalization can be uninformed about the
preference of the prediction network, undermining its performance. *Symmetrization* thus has the advantage in that it operates on the output of the backbone network and has access to the output of all group-transformed inputs, potentially leading to a wider variety of options and more informed choices.

The Frame Averaging (Puny et al. (2021)) approach is similar to the ones described in Basu et al. (2023) in the sense the it involves computing the output of a backbone network for multiple transformed inputs. *Frames* are a small subset of the whole possible set of group transformations, for which it holds that averaging over just the frame already results in equivariance or invariance. While this approach results in a smaller performance penalty, as requires less passes through the backbone, it only works if the correct frame could be selected for the given group and input, which is a non-trivial task. While theoretically it could be applied with existing pretrained backbones, results for this use case are not currently available.

## 4. Our contributions

The original paper improved on *equizero* and proposed *λ-equitune*, then validated them on an exceptionally wide range
of tasks in the domain of vision, natural language processing and reinforcement learning. Having results from such a
diverse set of tasks and using multiple backbone models is a strong testament to well-working methods. On the other
hand, we noticed that the publication included a different subset of the transfer learning methods for different tasks,
so we wanted to verify whether the results also hold for the missing experiments. The publication included a limited
discussion of the weight patterns *λ-equitune* learns, but it was based on a plot created for a single training example,
which does not generalize and is insufficient for drawing meaningful conclusions. Additionally, we also noticed
that many of the tasks chosen, for example image classification, were not equivariant but invariant in their nature, so
good results on these does not necessarily verify true equivariance of the solution. These observations motivated us to
perform reproducibility studies on some of the original data sets, expand the discussion of the inner workings of
*λ-equitune*, and to perform similar studies on additional data that tests the equivariant properties more.

In addition, we noticed that even the most sophisticated method proposed, *λ-equitune*, inspects each feature map
individually when calculating the weight, disregarding a significant source of information. Given the enormous success
of the Attention-based architectures in almost all areas of deep learning in recent years (Bommasani et al. (2021)), 
we hypothesized that using an Attention layer instead might provide better performance. 
This motivated us to create an extension of the original methods called *equiattention*.

The rest of the blog post will be structured accordingly to provide a summary of our methodologies and results:

- in 4.1 we discuss minor improvements we added to the original implementation
- section 4.2 explores whether the implementation was really equivariant
- in 4.3 we discuss our proposed alternative feature combination method
- 4.4 expands our understanding of the patterns *λ-equitune* learns
- in 4.5 the methods are tested on novel data sets
- and finally in 4.6 an extension of the original NLG tasks is described.

### 4.1 Reproducibility and minor implementation improvements

The authors kindly shared their implementation of the paper’s methods and experiments, which formed the basis of our
work. We started by reproducing the experiments related to image classification and we were pleased to find that we
could recreate Figure 4 from the original publication easily. Some training parameters were not specified in the
publication, in which case we used default values in the code base unless otherwise noted. However, upon closer
examination of the implementation, we discovered multiple points that we believe could be improved in the
implementation.

First, a softmax function was applied to the logits before passing them into the PyTorch `CrossEntropyLoss` function.
Since `CrossEntropyLoss` internally applies a softmax, this additional softmax acts as a smoothing function, hindering
the model’s ability to predict sharp distributions and slowing down training, as it decreases the gradients. Note that
after fixing this issue, our results are not directly comparable to the original implementation even at the same
learning rate due to this gradient magnitude difference.

Training was done in two phases: first only the weight network is trained, while keeping the backbone frozen (even the
layers that come after the weight network), which is referred to as pre-finetuning. Then the whole network is
fine-tuned, which is simply referred to as fine-tuning. The fine-tuning step is different from what the authors have
originally used, as in this step they kept the weight network frozen and only trained the backbone. We found no
theoretical justification for this approach, and also found it to perform worse in practice, so we kept the weight
network trainable during finetuning. We report results after each step to make comparisons easier with the original
publication, however we believe that results that only use pre-finetuning are more relevant and are more in line with
how a method such as *λ-equitune* would be used in practice. Especially when trying to achieve equivariance on a special
task like medical segmentation, where typically only limited training samples are available, keeping the backbone
network frozen and only training the weight network can seriously lower the risk of overfitting.

Our experiments show that removing the redundant softmax and adopting end-to-end finetuning significantly improve
performance. With these changes, along with using a lower learning rate of $5e-8$ for better training stability, we
achieve results that surpass those reported in the original paper. We achieved an increase of 3.70
percentage points (11.12%) in Top1 accuracy on CIFAR100 when using 90 degree rotations as the group transformations and
only training the weight network (pre-finetuning), as can be seen in the table below. The increase of 4.98 percentage
points (9.28%) is also significant and notable in case of full finetuning. A small increase in performance can be seen also when using flips.

|    | Method        | Architecture-Transformation        |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|---:|:--------------|:-----------------------------------|-----------------------:|--------------------:|
|  0 | Original Code | CLIP w RN50 - rot90 - *λ-equitune* |                  31.42 |               51.17 |
|  1 | Updated Code  | CLIP w RN50 - rot90 - *λ-equitune* |                  35.12 |               56.15 |
|  2 | Original Code | CLIP w RN50 - flip - *λ-equitune*  |                  37.07 |               54.04 |
|  3 | Updated Code  | CLIP w RN50 - flip - *λ-equitune*  |                  37.69 |               55.64 |

*Table 1: Image classification results using the author's original and our modified code base*

### 4.2 Introducing equivariance into the CLIP image classification experiments

Upon a closer inspection of the implementation of the EquiCLIP experiments, we also noticed an important discrepancy
between the equations described in the paper and the actual algorithm implemented in the codebase. While the paper
described *λ-equitune* by performing a group inverse transformation on the output of each separate backbone model before
averaging the feature maps, in practice the code implementation simply took an average of the logits calculated by each
backbone without any inverse operation. Please see the equations below for a precise comparison of the mathematics of
the paper and the code.

Equations described in the publication:

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

In a correspondence with the authors, they shared that this is because the image classification experiments do not
require equivariance, only invariance, and it was not even possible to apply the inverse group transformation to the
logits (the final outputs) of the backbone models in this case, as those have no spatial meaning anymore. While we
certainly agree with these observations, we were interested in understanding how the truly equivariant method (as
described in the paper) would perform, so proceeded to make the necessary changes to the code. It is at this point that
we would like to note that *λ-equitune* was not implemented as a single generic framework that could be applied as a
post-processing step on any backbone, but was copied and adapted for each experiment individually.

In order to test true equivariance in an image processing setting, we modified the implementation to only run each
backbone until the last convolutional layer, performed the inverse transformation, weight calculation and feature
combination on this spatial features, then passed the resulting combined feature map through the remaining layers of the
backbone network without changing them. We performed this modification using the CLIP model with the ResNet backends
only, as this model lended itself most easily to these changes. An overview of the original implementation and our
changes can be seen on the figure below. It is important to note that the feature maps for the group-transformed input
images are not just transformed versions of the feature map of the original image, so applying the inverse
transformation does not yield 4 identical feature maps.

![Architecture diagrams of a non-equivariant network,
*λ-equitune* using the original implementation and our version of it](images/architecture_diagrams.svg)
*Figure n: Architecture diagrams of a non-equivariant network, *λ-equitune* using the original implementation and our
version of it*

By applying these changes and testing with 90 degree rotations as the group transformation, we achieved an increase of
xx in Top1 accuracy on CIFAR100 when using 90 degree rotations as the group transformations and only training the weight
network (pre-finetuning), as can be seen in the table below. This underlines the fact that using a truly equivariant
version of *λ-equitune* outperforms the existing implementation even when tested on invariant tasks.
In (section x) we explore the performance of this method on truly equivariant tasks.

### 4.3 *equiattention*: Using Attention as a feature combination method

In the original work, the weights of features from $gx$ in the average are obtained independently for each $gx$. We see
this as a potential limitation, as such an approach is withholding potentially crucial information for determining the
significance of specific features.

We note that the only requirement for obtaining equivariant weights in the given setting is maintaining equivariance for
permutation of feature maps. Specifically,

```math
\boldsymbol\pi(f([\forall g \in G: \mathbf{M_1}(gx))]) = f(\boldsymbol\pi([\forall g \in G: \mathbf{M_1}(gx))]))
```

for an array permutation operator $\boldsymbol\pi$ and a function $f$ that produces an array of weights from an array of
features. In this case, $f$ must be a permutation equivariant function.

One permutation equivariant transformation that is being successfully applied across modalities is 
Attention (Bahdanau et al 2014, Vaswani et al 2017, Dosovitskiy et al 2020). 
For this reason, we attempt to improve the results further by utilizing an attention-based framework 
for computing the weights of feature maps.

We hypothesize that allowing the weighting function to access information from all views of the input at once, 
rather than individually, will increase the flexibility of the weighting component. 
However, to stay closer to the original work of *λ-equitune* and simplify the task of the learned component, 
we enforce the output to be a linear combination of the initial feature maps. 

We derive our non-masked single head self-attention from the original formulation of (Vaswani et al 2017):

```math
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

for feature maps $H = [h_0, h_1, \dots, h_{|G|-1}]$ (labeled $h$ for hiddens) and arbitrary index $i
\in [0, |G|-1]$. $h_i$ is a feature map obtained by $\mathbf{M_1}(g_ix)$, the part of the backbone before projection layers.

We calculate queries and keys with unconstrained MLPs $QNet$ and $KNet$. 
The networks have the same structure between each other, and have to be adapted for the backbone encoder of choice, 
as the dimension of hiddens $h_i$ varies between models.

```math
Q_i = QNet(h_i)
```

```math
K_i = KNet(h_i)
```

```math
V_i = g_i^{-1}h_i
```

The values $V$ are obtained without an MLP. Instead, we only apply the inverse transformation to the inputs. 
This design choice guarantees that the output is a linear combination of (group-transformed) feature maps, 
and allows the use of a smaller feature combination model.

Using the Attention as described above, we can calculate the final output of *equiattention* as follows:

```math
\mathbf{M}_G^A(x) = \mathbf{M}_2(\frac{1}{|G|}\sum_{g \in G}^{|G|} \text{Attention\_module}([\mathbf{M_1}(g_0x), \dots, \mathbf{M_1}(g_{|G|-1}x)]))
```

where `Attention_module` takes the features sets and applies one attention operation with the aforementioned $Q$, $K$, $V$.

From the results, we observe that the described method of *equiattention* is on par with 
the feature-equivariant version of *λ-equitune*, the method which it directly extends.

|    | Method                | Architecture-Transformation    |   Prefinetune Top1 Acc |
|---:|:----------------------|:-------------------------------|-----------------------:|
|  0 | equivariant equitune  | CLIP w RN50 - rot90            |                  40.95 |
|  1 | equivariant attention | CLIP w RN50 - rot90            |                  40.65 |

Investigating the results, we find that both models consistently predict the maximum weight of 1 to one 
feature map, and 0 weight to all other views. For the *rot90* group, one that contains 4 possible views, 
the conventionally oriented view is chosen in about 75% of the cases.
We thus conclude that *equiattention* did not outperform *λ-equitune* because the latter is already confident and close to the optimum.

### 4.4 Visualizations: understanding what *λ-equitune* (and *equiattention*) learns

Add visualizations here

### 4.5 Replicability: verifying the effectiveness on new problems

Replicating the results on novel datasets which exhibit different properties is an important step in verifying the
effectiveness of any new method. This is why, instead of reproducing all results from the original publication, we
decided to perform replication on 2 new data sets: the ISIC2018 image classification and an extended version of the Natural Language Generation task.

#### 4.5.1 ISIC 2018 Medical Imaging dataset

Image classification via widely used benchmarks like ImageNet and CIFAR provides a helpful understanding of the
performance of the methodologies, as it places the result within the context of the multitude of other methods that have
been tested on the same datasets. However these images have a natural orientation, so making models trained on them
equivariant is less important. This is why we chose to test on a medical imaging dataset, where a natural orientation
does not exist and any rotations of the inputs are equally likely, and equivariance of the detections is important.
Thus, we use the ISIC 2018 dataset, which was published by the International Skin Imaging Collaboration (ISIC) as a
large-scale dataset of dermoscopy images.The dataset consists from 10015 training images and 194 validation images
belonging to seven distinct diagnostic categories: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic
keratosis / Bowen’s disease (intraepithelial carcinoma), Benign keratosis (solar lentigo / seborrheic keratosis / lichen
planus-like keratosis), Dermatofibroma, and Vascular lesion. An example of the dataset images is shown below:

![Sample of classes in ISIC 2018 data set](images/isic2018_sample.png)

Image classification was performed by finetuning CLIP with a Resnet 50 backbone. It can be seen in the results
that [whichever method works better - add results].

|   | Method               | Architecture-Transformation          |   Prefinetune Top1 Acc |   Finetune Top1 Acc |
|--:|:---------------------|:-------------------------------------|-----------------------:|--------------------:|
| 0 | Original Code        | CLIP w RN50 - rot90 - *λ-equitune*   |                  15.03 |               63.73 |
| 1 | Updated Code         | CLIP w RN50 - rot90 - *λ-equitune*   |                  16.58 |               64.77 |
| 2 | equivariant equitune | CLIP w RN50 - rot90                  |                  16.58 |               40.93 |

*Table n: Image classification results using the author's original and our modified code base on the ISIC 2018 medical
dataset*

#### 4.5.2 Extended NLG task

Additionally, the authors formalized a group-theoretic approach to fairness in Natural Language Generation (NLG) task. Previous work has shown that Large Language Models (LLMs), such as GPT-2, are biased towards certain demographic groups in their generations (Sheng et al, 2019; Prates, Avelar, and Lamb, 2020; Henderson et al, 2018). While there was notable effort put into evaluating bias in LLMs (Sheng et al, 2019; Nadeem, Bethke, and Reddy 2021; Abid, Farooqi, and Zou, 2021), little has been done to theoretically study the mitigation of this bias and allow for a generalizable approach.

Basu et al (2023) introduced a novel approach to fairness in NLG using group theory. Given a demographic group $D$ (e.g. [man, woman]) and a language model $M$ (e.g. GPT2) with vocabulary $\mathcal{V}$, the authors first define the lists $\mathcal{E}$, $\mathcal{N}$, and $\mathcal{G}$ of equality, neutral, and general words, respectively (full details of the meaning of these lists is found in the Appendix). Then they let $d$ be the size of demographic group $D$ and define a cyclic group $G = \{ e, g, ..., g^{d-1}\}$ with generator $g$. The group action of $G$ makes a right cyclic shift by one to the words in $\mathcal{E}$ (essentially swapping the demographic identifier) and does not affect the words in $\mathcal{N}$ or $\mathcal{G}$. For example, if $\mathcal{E}$ = [[man, woman]], then $g\mathcal{E}$ = [[woman, man]]. 

Furthermore, they define context $X$ as a sentence consisting of words in $\mathcal{V}$ and transformed context $gX$ to be the sentence that is a result of applying $g$ to each word in $X$.  For instance, if $X$ = *"The man worked as a"*, then $gX$ = *"The woman worked as a"*. Finally, the model $M$ is given a context $X_1$ and is asked to generate a continuation $X_2$. The authors call $M$ *group-theoretically fair* if:

```math
\forall g \in G: P(gX_2 | gX_1) = P(X_2 | X_1)
```

where $P(X_2 | X_1)$ is the probability of generating the continuation $X_2$ given the context $X_1$. 

Using their proposed *equitune* language model, *EquiLM*, the authors formally satisfy this property and demonstrate that their methods can reduce bias in the generated text, making the generations more fair.

However, in their work, the authors only focused on establishing fairness across binary demographic groups -- specifically: man vs. woman, Black vs. White, and straight vs. gay. These binary groups, while useful for initial studies, do not capture the full complexity of real-world demographics. In our extension of this work, we aim to explore whether the fairness improvements seen in binary groups also apply to
non-binary groups. All of the three considered demographic groups naturally extend beyond binary classifications. We extend the theoretical framework to work with such groups and test the results on the extended race axis, namely Black vs. White vs. Asian. By extending the fairness framework to non-binary groups, we can better reflect the diversity of human identities and ensure that the proposed methods can mitigate bias in real-world settings.

The figure below illustrates the difference in a) the standard GPT2 model, b) original EquiGPT2 and c) our extended version of EquiGPT2 for non-binary groups.

<p align="center"> <img src="images/gpt2-a-jpeg.jpg" width="372"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="images/equigpt_binary-b-jpeg.jpg" width="500"> </p>

<p align="center"> <img src="images/inputA-c-jpeg.jpg" width="600"> </p>

In the standard case of GPT2, the model is given an input and outputs the logits over the vocabulary. In EquiGPT2, the group action $g$ is applied to the input and an inverse transformation is applied to the output logits, which are then aggregated for the final output. Finally, in our extension, the size of the demographic group is 3, so we also apply $g^2$ and $g^{-2}$ appropriately (however, the approach generalizes to an arbitrary demographic group size).

It is worth noting that in their implementation, the original authors were, in fact, applying a 'forward' transformation $g$ to the output logits in EquiGPT2, rather than $g^{-1}$. In the binary setting they considered, this is still a valid approach, as in that case $g = g^{-1}$. However, this approach breaks for $d > 2$. Therefore, we adapt the approach to work for non-binary groups and derive a mathematical proof (see Apendix) showing that applying $g^{-1}$ to the output logits ensures the equivariance property (i.e. $M(gx) = gM(x),  \forall g \in G$) and satisfies the group-theoretic fairness property (i.e. $P(gX_2 | gX_1) = P(X_2 | X_1), \forall g \in G$).

The following figure shows the reproduced results from the original paper for the demographic group [Black, White] for GPT2 and EquiGPT2, as well as for the case of the extended demographic group [Black, White, Asian]. The figure underneath is the respective figure from the Race-Respect task, for comparison purposes. 

<p align="center"> <img src="images/race-respect-extension.png" width="500">  <img src="images/race-respect-original.jpeg" width="500"> </p>

In the Figures, the y-axis represents the ratio of "regard scores" on 500 examples generated by GPT2 and EquiGPT2. These scores were assigned using the models of Sheng et al. (2019), and particularly "regard_v2", which is an ensemble of three BERT models that take the majority label of the model runs. The classifiers are trained to assign "regard" labels ranging from negative (-1) to positive (1) which assess the bias on the LLM's sentence generations starting with prompts such as "The white/black/asian person was regarded as...".
The label "other" corresponds to cases where there was a label produced but the generated sentence was missing. 

Our reproduction results very closely resemble the original findings. EquiGPT2 results in a more equal distribution of positive and negative labels, in particular when it comes at the variable "Black". 
Regarding the extension of inreasing the size to non-binary groups, it was observed that the label distribution did not vary significantly in the case where the inverse transformation was correctly applied to the logits, compared to when the 'forward' transformation was applied on the output logits, as originally implemented. 
Additionally, the results indicate that for the variable "Asian" there is only a very slight but present shift towards equal distribution of positive and negative labels. 


## 5. Summary

Equivariant fine-tuning of large foundational models is an attractive approach for combining the unprecedented
representation capabilities of these models with the introduction of guaranteed equivariance required in certain
domains. We have reproduced, replicated and extended the work of Basu et al. (2023), where they introduced *equizero*  and
*λ-equitune* for performing such fine-tuning. We have achieved an increase of [xx pp] in top1 accuracy
on [whichever dataset] by improving code and parameters, a further increase of [xx pp] by improving the methodology, and
proposed a new method called *equiattention*, which performed on par with the best baseline. Additionally, we have
verified the efficacy of these methods on novel datasets that exhibit equivariant properties and delivered
visualizations to better understand the operation of the trained *λ-equitune* and *equiattention* methods. Overall, we
found these methods to be an interesting family of approaches that are worth further exploration, and we hope our work
contributed to the understanding of their strengths and weaknesses.

## 6. Future work

Due to time and computational constrains, we were not able to conduct an extensive hyperparameter and 
model architecture search for the described methods. Specifically, the attention-based method was only tested 
with a single attention layer, and without a designated CLS token. Additionally, we note that training is 
heavily dependent on learning rate, failing to change the model's prediction with SDG lr less than 0.05 and 
leading to numerical issues for wide range of learning rates with Adam optimizer (Kingma et al 2014).

## 7. Acknowledgements
We would like to thank the authors for making their code available and for their fast and detailed responses to our
inquiries. We would also like to thank Yongtuo Liu for his supervision of our work.

## 8. Individual contributions

All authors continuously contributed to the project and group discussions. In particular, 

**Mikhail Vlasenko** found and fixed the code discrepancies, suggested, implemented and described the *equiattention* method

**Ádám Divák** created the lambda weight visualizations for CLIP, wrote the backbone of and edited the blog post and the proposal

**Iason Skylitsis** delivered the ISIC 2018 medical dataset extension and orchestrated training runs on Snellius

**Milan Miletić** co-authored the NLG extension, wrote and created diagrams for the relevant parts of the blog post

**Zoe Tzifa-Kratira** co-authored the NLG extension, worked on the introduction, background, and medical imaging section of the blog post

## 9. References

Basu, S., Katdare, P., Sattigeri, P., Chenthamarakshan, V., Driggs-Campbell, K., Das, P., & Varshney, L. R. (2023). Efficient Equivariant Transfer Learning from Pretrained Models. http://arxiv.org/abs/2305.09900

Basu, S., Sattigeri, P., Ramamurthy, K. N., Chenthamarakshan, V., Varshney, K. R., Varshney, L. R., & Das, P. (2023). Equi-Tuning: Group Equivariant Fine-Tuning of Pretrained Models. www.aaai.org

Bekkers, E. J., Lafarge, M. W., Veta, M., Eppenhof, K. A., Pluim, J. P., & Duits, R. (2018). Roto-Translation Covariant Convolutional Networks for Medical Image Analysis.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, R., Chatterji, N., Chen, A., Creel, K., Davis, J. Q., Demszky, D., … Liang, P. (2021). On the Opportunities and Risks of Foundation Models. http://arxiv.org/abs/2108.07258

Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., & Halpern, A. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC).

Cohen, T., Geiger, M., & Weiler, M. (2018). A General Theory of Equivariant CNNs on Homogeneous Spaces. http://arxiv.org/abs/1811.02017

Cohen, T. S., & Welling, M. (2016). Group Equivariant Convolutional Networks.

Hoogeboom, E., Satorras, V. G., Vignac, C., & Welling, M. (2022). Equivariant Diffusion for Molecule Generation in 3D.

Kaba, S.-O., Mondal, A. K., Zhang, Y., Bengio, Y., & Ravanbakhsh, S. (2022). Equivariance with Learned Canonicalization Functions. http://arxiv.org/abs/2211.06489

Lafarge, M. W., Bekkers, E. J., Pluim, J. P. W., Duits, R., & Veta, M. (2021). Roto-translation equivariant convolutional networks: Application to histopathology image analysis. Medical Image Analysis, 68, 101849. https://doi.org/10.1016/j.media.2020.101849

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation Applied to Handwritten Zip Code Recognition. Neural Computation, 1(4), 541–551. https://doi.org/10.1162/neco.1989.1.4.541

Mondal, A. K., Panigrahi, S. S., Kaba, S.-O., Rajeswar, S., & Ravanbakhsh, S. (2023). Equivariant Adaptation of Large Pretrained Models.

Moskalev, A., Sepliarskaia, A., Bekkers, E. J., & Smeulders, A. (2023). On genuine invariance learning without weight-tying.

OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M., Belgum, J., … Zoph, B. (2023). GPT-4 Technical Report.

Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E. J., & Lipman, Y. (2021). Frame Averaging for Invariant and Equivariant Network Design.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., … Scialom, T. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5(1), 180161. https://doi.org/10.1038/sdata.2018.161

Tunyasuvunakool, K., Adler, J., Wu, Z., Green, T., Zielinski, M., Žídek, A., Bridgland, A., Cowie, A., Meyer, C., Laydon, A., Velankar, S., Kleywegt, G. J., Bateman, A., Evans, R., Pritzel, A., Figurnov, M., Ronneberger, O., Bates, R., Kohl, S. A. A., … Hassabis, D. (2021). Highly accurate protein structure prediction for the human proteome. Nature, 596(7873), 590–596. https://doi.org/10.1038/s41586-021-03828-1

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. http://arxiv.org/abs/1706.03762

Villar, S., Hogg, D. W., Storey-Fisher, K., Yao, W., & Blum-Smith, B. (2021). Scalars are universal: Equivariant machine learning, structured like classical physics.
