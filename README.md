# (Even More) Efficient Equivariant Transfer Learning from Pretrained Models

### Mikhail Vlasenko, Ádám Divák, Iason Skylitsis, Milan Miletić, Zoe Tzifa-Kratira

*Equivariance in deep learning refers to a model's ability to maintain consistent output changes in response to specific transformations of the input, ensuring that the model's behavior aligns predictably with the symmetries in the data. Many problems are known to be equivariant in nature, thus using a method that inherently has this inductive bias can increase the robustness and generalization capabilities of the models used. Several very large foundation models have been trained recently in multiple modalities, which deliver unprecedented performance in a wide variety of downstream tasks. These models, however, are not equivariant by their design, which limits their usability in contexts where this would be necessary. Re-training foundation models from scratch using an equivariant architecture is prohibitively expensive for most researchers, which is why several methods were proposed to get provably equivariant output from non-equivariant backbone architectures. We set out to explore the methods λ-equitune and equizero proposed by Basu et al., which were shown to deliver good results in a wide variety of downstream tasks. We perform replication studies, suggest code and parameter improvements that deliver significantly better results, and propose a new alternative method that we call equiattention. Additionally, we explore the performance of these methods on new problems and produce visualizations to better understand their working mechanisms.*

-----

This repository contains a reproduction and extension of ["Efficient Equivariant Transfer Learning from Pretrained Models"](https://arxiv.org/abs/2305.09900) by Basu et al. (2023).

Please read [Blogpost.md](Blogpost.md) for the full article, containing detailed information on our reproduction experiments and extension study.

## Conda Environment
First create the required conda environment, activate it, and install clip, Imagenet_V2 as follows
```
conda env create -f environment.yml
conda activate lambda_equitune
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```

## How to reproduce

All our experiments are tracked using [Weights and Biases](https://docs.wandb.ai/). To set it up correctly, follow these steps:

1. **Modify the [.env](.env) File**:
   - Add your `entity name` (your username or organization name).
   - Add the `project name` you want for the project.

2. **Log in to Weights and Biases**:
   Before running any experiment, log in and provide your API key when prompted:
   ```sh
   wandb login
   ```
3. **Reproduce Initial Experiments**:
   - Run the job file to reproduce the original author's zeroshot results that correspond to Figure 4 in the original paper:
      ```sh
      sbatch job_files/reproduce_bar_plots.job
      ```
   - Plot the results using the provided scripts:
      ```sh
        python demos/plot_results.py
        python demos/plot_results2.py
      ```
4. **Reproduce Table 1 from the Blogpost**: 
    - Run the following job file:
      ```sh
      sbatch job_files/compare_original_updated_cifar.job
      ```
    - Create the table by running the following jupyter notebook: [demos/original_vs_updated_cifar.ipynb](demos/original_vs_updated_cifar.ipynb)

5. **Reproduce Table 3 from the Blogpost**: 
    - Run the following job file:
      ```sh
      sbatch demos/equivariant_equitune_vs_attention.ipynb
      ```
    - Create the table by running the following jupyter notebook: [demos/equivariant_equitune_vs_attention.ipynb](demos/equivariant_equitune_vs_attention.ipynb)

6. **Reproduce Table 4 from the Blogpost**: 
    - Run the following job file:
      ```sh
      sbatch job_files/compare_original_updated_isic.job
      ```
    - Create the table by running the following jupyter notebook: [demos/original_vs_updated_isic.ipynb](demos/original_vs_updated_isic.ipynb)


If you find the code useful, please cite it as
```
@misc{vlasenko2024efficient,
  title={(Even More) Efficient Equivariant Transfer Learning from Pretrained Models},
  author={Mikhail Vlasenko and Ádám Divák and Iason Skylitsis and Milan Miletić and Zoe Tzifa-Kratira},
  year={2024},
  note={Work in progress}
}
```
