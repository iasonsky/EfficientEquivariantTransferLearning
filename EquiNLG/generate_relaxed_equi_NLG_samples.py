import argparse
import torch

from wordlists import general_words
from relaxed_equi_neural_text_decoding import generate_relaxed_equi_text
from exp_utils import THE, SPACE, BLACK, WHITE, ASIAN, MAN, WOMAN, GAY, STRAIGHT, KNOWN, DESCRIBED, REGARDED, THOUGHT,\
    WELL_KNOWN, WORKED, JOB, PART_TIME, EARNED, STARTED_WORKING

device = "cuda" if torch.cuda.is_available() else "cpu"

gender_equality_words_sets = [[' man', ' woman']]
color_equality_words_sets = [[' Black', ' White']]
sexuality_equality_words_sets = [[' gay', ' straight']]


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
                    context = THE + SPACE + args.demographic_group + SPACE + respect_bias_context
                    args.context = context
                    sample = generate_relaxed_equi_text(args)
                    f.write(sample + '\n')
        elif args.bias_context == 'occupation':
            for seed in range(args.seed_low, args.seed_high):
                args.seed = seed
                for occupation_bias_contexts in occupation_contexts:
                    context = THE + SPACE + args.demographic_group + SPACE + occupation_bias_contexts
                    args.context = context
                    sample = generate_relaxed_equi_text(args)
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

    args = parser.parse_args()

    # set general words and equality words
    args.general_words = general_words
    if args.demographic_group in ["MAN", "WOMAN"]:
        args.equality_word_sets = gender_equality_words_sets
    elif args.demographic_group in ["BLACK", "WHITE", "ASIAN"]:
        args.equality_word_sets = color_equality_words_sets
    elif args.demographic_group in ["GAY", "STRAIGHT"]:
        args.equality_word_sets = sexuality_equality_words_sets
    else:
        raise NotImplementedError

    # define the filename where the samples will be stored
    dir = "generated_samples/relaxedEquiGPT2/"
    filepath = dir + args.bias_context + '_' + args.demographic_group.split(' ')[0] + '.txt'

    # generate samples
    generate_samples(args, dir, filepath)
