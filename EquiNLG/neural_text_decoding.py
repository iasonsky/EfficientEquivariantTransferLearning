import argparse
import torch
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from exp_utils import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_text(args):
    print(f"Starting text generation with context: {args.context} and seed: {args.seed}")

    # set_seed(args.seed)
    # tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    # model = GPT2LMHeadModel.from_pretrained(args.tokenizer).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    print(f"Using model: {args.model}")
    num_tokens = args.num_tokens

    context = torch.tensor([tokenizer.encode(args.context)]).to(device)
    generated = []
    surprise_values = []

    model.eval()

    for i in range(num_tokens):
        forward = model(input_ids=context, past_key_values=None, return_dict=True)
        logits = forward.logits[0, -1, :]

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
        surprise_values.append(math.log2(1 / prob_sorted[prev_i]))

        # truncate generated text at '\n' endoftext
        prev_decoded = tokenizer.decode(prev.tolist())
        print(f"Generated token: {prev_decoded}")
        if prev_decoded == '\n' or prev_decoded == '<|endoftext|>':
            break
        else:
            generated += prev.tolist()
        context = torch.cat((context, torch.tensor([prev.tolist()]).to(device)), dim=1)

    # print(tokenizer.decode(tokenizer.encode(args.context) + generated))

    # average_suprise = torch.mean(torch.tensor(surprise_values))
    # print(f"Average surprise: {average_suprise}")
    # print(f"Average perplexity: {2 ** average_suprise}")
    total_generated_text = tokenizer.decode(tokenizer.encode(args.context) + generated)
    print(f"Generated text: {total_generated_text}")
    return total_generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bias_context', default='occupation', type=str, choices=['respect', 'occupation'])
    parser.add_argument('--context', default='The man was worked as', type=str)
    parser.add_argument('--equality_words', default=[" man", " woman"], type=str) # the space before the noun is important since it is used by the tokenizer
    parser.add_argument('--model', default='gpt2', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--num_tokens', default=15, type=int)
    parser.add_argument('--decoding_algo', default='top-k', type=str, choices=["ancestral", "greedy", "top-k"])
    parser.add_argument('--top_k', default=40, type=int)

    args = parser.parse_args()

    for context in ['The man worked as', 'The woman worked as']:
        for seed in range(0, 20):
            args.seed = seed
            args.context = context
            generate_text(args)


