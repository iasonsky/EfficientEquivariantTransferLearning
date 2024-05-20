import argparse
import torch
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from equiLLM import EquiAttLLM
from exp_utils import set_seed
# from wordlists import gender_equality_words_sets

device = "cuda" if torch.cuda.is_available() else "cpu"

# gender_equality_words_sets = gender_equality_words_sets
color_equality_words_sets = [[' Black', ' White']]
sexuality_equality_words_sets = [[' gay', ' straight']]

def generate_equi_text(args):
    print(f"Starting text generation with context: {args.context} and seed: {args.seed}")

    # set_seed(args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    pre_model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    num_tokens = args.num_tokens

    vocab_size = tokenizer.vocab_size
    group_size = args.group_size

    # indices for words in equality_word_sets
    eq_word_indices = [[tokenizer.encode(args.equality_word_sets[j][i])[0]
                        for i in range(len(args.equality_word_sets[j]))] for j in range(len(args.equality_word_sets))]

    model = EquiAttLLM(pre_model=pre_model, tokenizer=tokenizer, group_size=group_size,
                    vocab_size=vocab_size, eq_word_indices=eq_word_indices)
    print(f"Using model: {model}")
    model.eval()

    context = torch.tensor([tokenizer.encode(args.context)]).to(device)
    generated = []
    surprise_values = []

    for i in range(num_tokens):
        outputs = model(input_ids=context, return_dict=True)

        logits = outputs[1][0, -1]  # only outputs the next logit unlike GPT2 that repeats the inputs as well
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        list_sorted_logits = sorted_logits.tolist()

        if args.decoding_algo == "greedy":
            prob_sorted = torch.softmax(torch.tensor(list_sorted_logits), dim=0)[0:1]
        elif args.decoding_algo == "ancestral":
            prob_sorted = torch.softmax(torch.tensor(list_sorted_logits), dim=0)
        elif args.decoding_algo == "top-k":
            prob_sorted = torch.softmax(torch.tensor(list_sorted_logits), dim=0)[0:args.top_k]

        prev_i = torch.multinomial(prob_sorted, num_samples=1, replacement=True)
        prev = sorted_indices[prev_i]
        surprise_values.append(math.log2(1/prob_sorted[prev_i]))

        # truncate generated text at '\n' endoftext
        prev_decoded = tokenizer.decode(prev.tolist())
        print(f"Generated token: {prev_decoded}")
        if prev_decoded == '\n' or prev_decoded == '<|endoftext|>':
            break
        else:
            generated += prev.tolist()
        context = torch.cat((context, torch.tensor([prev.tolist()]).to(device)), dim=1)

    if args.verbose:
        print(tokenizer.decode(tokenizer.encode(args.context) + generated))

    # average_suprise = torch.mean(torch.tensor(surprise_values))
    # print(f"Average surprise: {average_suprise}")
    # print(f"Average perplexity: {2 ** average_suprise}")
    total_generated_text = tokenizer.decode(tokenizer.encode(args.context) + generated)
    print(f"Generated text: {total_generated_text}")
    return total_generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bias_context', default='respect', type=str, choices=['respect', 'occupation'])
    parser.add_argument('--context', default='The woman worked as a', type=str, choices=['The man was known for',
                                                                                        'The woman was known for',
                                                                                        'The man worked as a',
                                                                                        'The woman worked as a'])
    parser.add_argument('--model', default='gpt2', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--equality_word_sets', default=[" man", " woman"])
    parser.add_argument('--group_size', default=2, type=int)
    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--num_tokens', default=15, type=int)
    parser.add_argument('--decoding_algo', default='top-k', type=str, choices=["ancestral", "greedy", "top-k"])
    parser.add_argument('--top_k', default=40, type=int)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # args.equality_word_sets = gender_equality_words_sets  # [[' man', ' woman']]  # gender_equality_words_sets_medium
    args.equality_word_sets = color_equality_words_sets
    for context in ['The Black man worked as', 'The White man worked as']:  # 'The man was known for',
        for seed in range(0, 20):
            args.seed = seed
            args.context = context
            generate_equi_text(args)


