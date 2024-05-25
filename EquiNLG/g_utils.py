import torch
from operator import itemgetter


def cyclic_group_generator(vocab_size, group_size, eq_indices):
    """
    :param vocab_size: size of the vocab
    :param group_size: size of the group
    :param eq_indices: set of  list of indices of the tokens for the equivariant words in the vocab; each list of indices of the same size as group size
    :return: a group generator of the required cyclic group consisting of all the equivariant word indices
    """
    g = {i: i for i in range(vocab_size)}  # generator initialized as id

    for i in range(group_size):
        next_group_element = (i + 1) % group_size
        # g[eq_indices[i]] = eq_indices[next_group_element]
        for j in range(len(eq_indices)):
            g[eq_indices[j][i]] = eq_indices[j][next_group_element]

    g['___size___'] = group_size  # add length of the group as a value
    return g


def cyclic_group(g, vocab_size, group_size):
    """
    :param g: cyclic group generator
    :param group_size: size of the group
    :return: return a list of elements of a cyclic group
    """
    # add id to the group G
    G = [{i: i for i in range(vocab_size)}]

    for i in range(group_size - 1):
        # apply the generator repeatedly to obtain the entire group
        curr_g = G[-1]
        next_g = {i: g[curr_g[i]] for i in range(vocab_size)}
        G.append(next_g)

    return G


def g_transform_data(data, G, device):
    '''
    :param data: any tensor data of input on which group is applied
    :param G: set of group elements
    :return: list of transformed data for equituning
    '''
    # print("Debugging function: g_transform_data")
    # print("  Group Elements:", G)

    data_shape = data.size()
    untransformed_data = data.view(-1)
    transformed_data = [untransformed_data]

    for i in range(len(G)-1):
        curr_g = G[i+1]
        current_data = torch.tensor(itemgetter(*(untransformed_data.tolist()))(curr_g), device=device)
        transformed_data.append(current_data)
        # print(f"  After applying group element {i}: {current_data}")

    transformed_data = torch.stack(transformed_data).view(len(G), data_shape[0], data_shape[1])
    transformed_data.to(device)

    return transformed_data


def g_inv_transform_prob_data(data_list, G):
    '''
    Note: Group actions are on batch_size x |V|, instead of batch_size x 1
    :param data: any tensor data
    :param g: group generator
    :return: list of transformed data for equituning
    '''
    # print("Debugging function: g_inv_transform_prob_data")
    output_data_list = data_list.clone()  # dim [group_size, batch_size, num_tokens, |V|]
    g_indices = []
    for g in G:
        # Define the inverse transformation
        g_inv = {val: key for key, val in g.items()}
        g_inv = dict(sorted(g_inv.items()))
        
        g_index = [g_inv[i] for i in range(len(g_inv))]
        g_indices.append(g_index)

    # print("  Initial data list for inverse transformation:")
    # print(data_list)

    for i in range(len(data_list)):  # iterate over group size
        output_data_list[i, :, :, g_indices[i]] = output_data_list[i, :, :, :].clone()

    # print("  Final data list after inverse transformation:")
    # print(output_data_list)
    return output_data_list


def g_transform_prob_data(data_list, G, group_index=1):
    '''
    Note: Group actions are on batch_size x |V|, instead of batch_size x 1
    :param data: any tensor data
    :param g: group generator
    :return: list of transformed data for equituning
    '''
    output_data_list = data_list.clone()
    g_indices = []
    for g in G:
        g_index = [g[i] for i in range(len(g))]
        g_indices.append(g_index)

    for i in range(len(data_list)):
        output_data_list[:, :] = data_list[:, g_indices[group_index]].clone()

    return output_data_list


def g_inv_transform_prob_data_new(data_list, G, sequence_len, vocab_size):
    output_data_list = data_list.clone()
    
    return output_data_list.view(-1, sequence_len, vocab_size)
