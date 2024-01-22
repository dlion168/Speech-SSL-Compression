# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from typing import Optional, Tuple

import numpy as np
import torch

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def compute_block_mask_1d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:

    B, L = shape

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(L / mask_length)

        inp = torch.zeros((B, 1, sz))
        w = torch.ones((1, 1, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(sz * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose1d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > L:
            mask = mask[..., :L]

    else:
        mask = torch.zeros((B, L))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length)
                    * (1 + mask_dropout)
                ),
            ),
        )

        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [])

        offset = mask_length // 2
        for i in range(mask_length):
            k1 = i - offset
            inds[0].append(centers[0])
            inds[1].append(centers[1] + k1)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)

        mask[(i0, i1)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv1d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.ones((1, 1, 3))
        w[..., 1] = 0
        all_nbs = get_nbs(B, mask, w)

    mask = mask.view(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.unsqueeze(0), w).squeeze(0)

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask

def get_buckets(sizes, num_buckets):
    buckets = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation="lower",
        )[1:]
    )
    return buckets

def get_bucketed_sizes(orig_sizes, buckets):
    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes