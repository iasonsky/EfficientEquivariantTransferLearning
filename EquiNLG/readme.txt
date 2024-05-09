### Fairness in natural language generation: EquiNLG
Codes to be run from the EquiNLG folder

To generate text simply using GPT2 for the demographic group "**man**" (choose from ["man", "woman", "Black", "White", "straight", "gay"]) for the bias context "**respect**" (choose from ["respect", "occupation"]), run the following
```commandline
python generate_NLG_samples.py --demographic_group MAN --bias_context respect 
```

To generate text using R-EquiGPT2 for the demographic group "**man**" (choose from ["man", "woman", "Black", "White", "straight", "gay"]) for the bias context "**respect**" (choose from ["respect", "occupation"]), run the following
```commandline
python generate_relaxed_equi_NLG_samples.py --demographic_group MAN --bias_context respect 
```

To generate text using EquiGPT2 for the demographic group "**man**" (choose from ["man", "woman", "Black", "White", "straight", "gay"]) for the bias context "**respect**" (choose from ["respect", "occupation"]), run the following
```commandline
python generate_equi_NLG_samples.py --demographic_group MAN --bias_context respect 
```