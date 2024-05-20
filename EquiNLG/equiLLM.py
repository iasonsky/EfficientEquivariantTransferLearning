import torch.nn as nn
import random
import torch

from g_utils import cyclic_group_generator, cyclic_group, g_transform_data, g_inv_transform_prob_data, g_inv_transform_prob_data_new
from torch.nn import CrossEntropyLoss
from attention_aggregation import AttentionAggregation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class EquiLLM(nn.Module):
    def __init__(self, pre_model, tokenizer, group_size=2, vocab_size=8, eq_word_indices=[2, 7],
                 feature_extracting=False, group_type='cyclic'):
        super(EquiLLM, self).__init__()
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.eq_word_indices = eq_word_indices

        in_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)
        out_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)

        self.in_G = cyclic_group(g=in_g, vocab_size=vocab_size, group_size=group_size)
        self.out_G = cyclic_group(g=out_g, vocab_size=vocab_size, group_size=group_size)

        set_parameter_requires_grad(pre_model, feature_extracting)
        self.pre_model = pre_model
        self.tokenizer = tokenizer

        self.loss_fn = CrossEntropyLoss()

    def compute_loss(self, logits, labels):
        """
        Args:
            logits: logits of dim [batch_size, num_tokens in a sentence, vocab_size]
            labels: logits of dim [batch_size, num_tokens in a sentence]
        Returns:
        """
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1)[:, :, 0:-1], labels[:, 1:])
        else:
            loss = 0
        return loss

    def forward(self, input_ids, return_dict=True, labels=None):
        transformed_context = g_transform_data(input_ids, self.in_G, device)  # dim: [|G|, batch_size, seq_length]
        #print(f"INPUT IDS: {input_ids}")
        #print(f"INPUT IDS SHAPE: {input_ids.shape}")
        #print(f"Transformed context: {transformed_context}")
        #print(f"Transformed context shape: {transformed_context.shape}")

        # get transformed outputs
        transformed_logits = []
        for i in range(len(transformed_context)):
            context = torch.tensor([transformed_context[i].tolist()]).to(device)
            forward = self.pre_model(input_ids=context, past_key_values=None, return_dict=return_dict)
            transformed_logits.append(forward.logits[0])  # forward.logits[0] dim [batch_size, seq_len, vocab_size]

        # inverse transform the texts corresponding to each of the transformed contexts
        transformed_logits = torch.stack(transformed_logits)
        #print(f"Transformed logits: {transformed_logits}")
        #print(f"Transformed logits shape: {transformed_logits.shape}")

        group_logits = g_inv_transform_prob_data(transformed_logits, G=self.out_G)
        # print(f"Group logits after inverse transformation: {group_logits}")

        logits = torch.mean(group_logits, dim=0, keepdim=False)  # dim [batch_size, seq_len, vocab_size]
        # print(f"Final logits: {logits}")

        loss = self.compute_loss(logits, labels)

        return [loss, logits]


class REquiLLM(nn.Module):
    def __init__(self, pre_model, tokenizer, group_size=2, vocab_size=8, eq_word_indices=[2, 7],
                 neutral_word_indices=[5, 6], feature_extracting=False, group_type='cyclic'):
        super(REquiLLM, self).__init__()
        self.vocab_size = vocab_size
        self.eq_word_indices = eq_word_indices
        self.neutral_word_indices = neutral_word_indices
        self.group_size = group_size
        # non_general words  = vocab - general_words
        self.non_general_word_indices = self.eq_word_indices + self.neutral_word_indices

        self.non_general_words_mask = self.get_token_mask()
        # multiply with group_size to adjust for averaging and masking
        self.general_words_mask = (1.0 - self.non_general_words_mask) * self.group_size

        in_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)
        out_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)

        self.in_G = cyclic_group(g=in_g, group_size=group_size, vocab_size=vocab_size)
        self.out_G = cyclic_group(g=out_g, group_size=group_size, vocab_size=vocab_size)

        set_parameter_requires_grad(pre_model, feature_extracting)
        self.pre_model = pre_model
        self.tokenizer = tokenizer

        self.loss_fn = CrossEntropyLoss()

    def get_token_mask(self):
        mask = torch.zeros(size=(self.vocab_size,))
        for i in range(len(mask)):
            mask[i] = 1 if i in self.non_general_word_indices else 0

        return mask

    def relaxed_mask_group_transformations(self, group_logits):
        """
        Args:
            group_logits: # dim [|G|, batch_size, seq_len, vocab_size]
        Returns:
        """
        identity_element_mask = self.non_general_words_mask + self.general_words_mask
        non_identity_element_mask = self.non_general_words_mask

        for i in range(len(group_logits)):
            mask = identity_element_mask if i == 0 else non_identity_element_mask
            mask = mask.to(device)
            group_logits[i] = torch.einsum("ijk, k -> ijk", group_logits[i].clone(), mask)

        return group_logits

    def compute_loss(self, logits, labels):
        """
        Args:
            logits: logits of dim [batch_size, num_tokens in a sentence, vocab_size]
            labels: logits of dim [batch_size, num_tokens in a sentence]
        Returns:
        """
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1)[:, :, 0:-1], labels[:, 1:])
        else:
            loss = 0
        return loss

    def forward(self, input_ids, return_dict=True, labels=None):
        transformed_context = g_transform_data(input_ids, self.in_G, device)  # dim: [|G|, batch_size, seq_length]

        # get transformed outputs
        transformed_logits = []
        for i in range(len(transformed_context)):
            context = torch.tensor([transformed_context[i].tolist()]).to(device)
            forward = self.pre_model(input_ids=context, past_key_values=None, return_dict=return_dict)
            transformed_logits.append(forward.logits[0])  # forward.logits[0] dim [batch_size, seq_len, vocab_size]

        # inverse transform the texts corresponding to each of the transformed contexts
        transformed_logits = torch.stack(transformed_logits)  # dim [|G|, batch_size, seq_len, vocab_size]
        group_logits = g_inv_transform_prob_data(transformed_logits, G=self.out_G)  # dim [|G|, batch_size, seq_len, vocab_size]
        group_logits = self.relaxed_mask_group_transformations(group_logits)
        logits = torch.mean(group_logits, dim=0, keepdim=False)  # dim [batch_size, seq_len, vocab_size]

        loss = self.compute_loss(logits, labels)

        return [loss, logits]


class EquiAttLLM(nn.Module):
    def __init__(self, pre_model, tokenizer, group_size=2, vocab_size=8, eq_word_indices=[2, 7],
                 feature_extracting=False, group_type='cyclic'):
        super(EquiAttLLM, self).__init__()
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.eq_word_indices = eq_word_indices

        in_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)
        out_g = cyclic_group_generator(vocab_size=vocab_size, group_size=group_size, eq_indices=eq_word_indices)

        self.in_G = cyclic_group(g=in_g, vocab_size=vocab_size, group_size=group_size)
        self.out_G = cyclic_group(g=out_g, vocab_size=vocab_size, group_size=group_size)

        set_parameter_requires_grad(pre_model, feature_extracting)
        self.pre_model = pre_model
        self.tokenizer = tokenizer

        self.loss_fn = CrossEntropyLoss()

    def compute_loss(self, logits, labels):
        """
        Args:
            logits: logits of dim [batch_size, num_tokens in a sentence, vocab_size]
            labels: logits of dim [batch_size, num_tokens in a sentence]
        Returns:
        """
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1)[:, :, 0:-1], labels[:, 1:])
        else:
            loss = 0
        return loss

    def forward(self, input_ids, return_dict=True, labels=None):
        transformed_context = g_transform_data(input_ids, self.in_G, device)  # dim: [|G|, batch_size, seq_length]
        #print(f"INPUT IDS: {input_ids}")
        #print(f"INPUT IDS SHAPE: {input_ids.shape}")
        #print(f"Transformed context: {transformed_context}")
        #print(f"Transformed context shape: {transformed_context.shape}")

        # get transformed outputs
        transformed_logits = []
        for i in range(len(transformed_context)):
            context = torch.tensor([transformed_context[i].tolist()]).to(device)
            forward = self.pre_model(input_ids=context, past_key_values=None, return_dict=return_dict)
            transformed_logits.append(forward.logits[0])  # forward.logits[0] dim [batch_size, seq_len, vocab_size]

        # inverse transform the texts corresponding to each of the transformed contexts
        transformed_logits = torch.stack(transformed_logits)
        #print(f"Transformed logits: {transformed_logits}")
        #print(f"Transformed logits shape: {transformed_logits.shape}")

        group_logits = g_inv_transform_prob_data(transformed_logits, G=self.out_G)
        # print(f"Group logits after inverse transformation: {group_logits}")
        group_size, batch_size, seq_len, vocab_size = group_logits.shape

        attention_aggregation = AttentionAggregation(group_size, vocab_size, seq_len).to(device)
        logits = attention_aggregation(group_logits) 
        #logits = torch.mean(group_logits, dim=0, keepdim=False)  # dim [batch_size, seq_len, vocab_size]
        # print(f"Final logits: {logits}")

        loss = self.compute_loss(logits, labels)

        return [loss, logits]


