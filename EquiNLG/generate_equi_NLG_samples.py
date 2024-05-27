import argparse
import torch

# from wordlists import gender_equality_words_sets
from equi_neural_text_decoding import generate_equi_text
from exp_utils import THE, SPACE, BLACK, WHITE, ASIAN, MAN, WOMAN, GAY, STRAIGHT, KNOWN, DESCRIBED, REGARDED, THOUGHT,\
    WELL_KNOWN, WORKED, JOB, PART_TIME, EARNED, STARTED_WORKING

device = "cuda" if torch.cuda.is_available() else "cpu"

# gender_equality_words_sets = gender_equality_words_sets
color_equality_words_sets = [[' Black', ' White', ' Asian']]
sexuality_equality_words_sets = [[' gay', ' straight']]


def arg_to_variable(arg):
    """
    Converts the input argument of type string to one of the global variables defined in this file.
    """
    if arg == 'BLACK':
        return BLACK
    elif arg == 'WHITE':
        return WHITE
    elif arg == 'ASIAN':
        return ASIAN
    elif arg == 'MAN':
        return MAN
    elif arg == 'WOMAN':
        return WOMAN
    elif arg == 'GAY':
        return GAY
    elif arg == 'STRAIGHT':
        return STRAIGHT
    

def generate_samples(args, dir, filepath):
    """
    writes samples for the given context and seed number to the appropriate .txt file
    Returns:
    """

    # define the various contexts
    respect_contexts = [KNOWN, DESCRIBED, REGARDED, THOUGHT, WELL_KNOWN]
    occupation_contexts = [WORKED, JOB, PART_TIME, EARNED, STARTED_WORKING]

    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(filepath, "w") as f:
        if args.bias_context == 'respect':
            for seed in range(args.seed_low, args.seed_high):
                args.seed = seed
                for respect_bias_context in respect_contexts:
                    context = THE + SPACE + arg_to_variable(args.demographic_group) + SPACE + respect_bias_context
                    args.context = context
                    sample = generate_equi_text(args)
                    f.write(sample + '\n')
        elif args.bias_context == 'occupation':
            for seed in range(args.seed_low, args.seed_high):
                args.seed = seed
                for occupation_bias_contexts in occupation_contexts:
                    context = THE + SPACE + arg_to_variable(args.demographic_group) + SPACE + occupation_bias_contexts
                    args.context = context
                    sample = generate_equi_text(args)
                    f.write(sample + '\n')

        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_low', default=0, type=int)
    parser.add_argument('--seed_high', default=1, type=int)
    parser.add_argument('--bias_context', default='respect', type=str, choices=['respect', 'occupation'])
    parser.add_argument('--demographic_group', default=BLACK, type=str)
    parser.add_argument('--model', default='gpt2', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--equality_word_sets', default=[" man", " woman"])
    parser.add_argument('--group_size', default=2, type=int)
    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--num_tokens', default=15, type=int)
    parser.add_argument('--decoding_algo', default='top-k', type=str, choices=["ancestral", "greedy", "top-k"])
    parser.add_argument('--top_k', default=40, type=int)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    print(f"Arguments: {args}")

    if args.demographic_group in ["MAN", "WOMAN"]:
        # args.equality_word_sets = gender_equality_words_sets
        pass
    elif args.demographic_group in ["BLACK", "WHITE", "ASIAN"]:
        args.equality_word_sets = color_equality_words_sets
    elif args.demographic_group in ["GAY", "STRAIGHT"]:
        args.equality_word_sets = sexuality_equality_words_sets
    else:
        raise NotImplementedError

    # define the filename where the samples will be stored
    dir = "generated_samples/equiGPT2/"
    filepath = dir + args.bias_context + '_' + args.demographic_group.split(' ')[0] + '.txt'

    print(f"Output directory: {dir}")
    print(f"Output file path: {filepath}")

    # generate samples
    generate_samples(args, dir, filepath)