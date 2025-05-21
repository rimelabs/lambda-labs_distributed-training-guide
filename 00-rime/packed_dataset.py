import datasets
import os
import torch
import time

from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def get_attention_mask_and_pos_ids_for_packed_sequence(x, eos_id=128_262, model_dtype=torch.bfloat16):
    # store sequence length in variable for easier readability
    T = x.size(0)
    
    # Manually set last token to be end-of-ai token to keep this code happy
    x[-1] = eos_id
    
    # get indices of all EOS tokens
    eos_indices = (x == eos_id).nonzero().squeeze()    
    # from indices, get length of each sequence
    reps = torch.cat([eos_indices[[0]]+1, eos_indices[1:] - eos_indices[:-1]])
    # repeat each eos index n times along dimension 1 (n is the number of tokens in the sequence)
    repeated_idx = torch.repeat_interleave(eos_indices, reps).view(1,-1).expand(T, -1)
    # create tensor with all indices from 0 to T-1 repeated T times along dimesion 1
    mask_indices = torch.arange(T).view(-1,1).expand(-1, T)
    # create causal mask and additionally mask out all tokens from preceeding sequences
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(-1, -1)
    
    # prepared inverted, additive bias mask (attend = 0.0; don't attend = ~ -Inf)
    mask.masked_fill_(mask_indices > repeated_idx, False)
    min_dtype = torch.finfo(model_dtype).min
    inv_mask = (mask.eq(False)).to(dtype=torch.bfloat16) * min_dtype

    # prepare position_ids
    pos_ids = torch.arange(T) - torch.repeat_interleave(torch.cat([torch.tensor([0]), eos_indices+1], dim=0)[:-1], reps)
    
    return inv_mask, pos_ids

def pad_and_collate(data):
    inv_mask_and_pos_ids = [ get_attention_mask_and_pos_ids_for_packed_sequence(d['input_ids']) for d in data ]

    return {
        "input_ids" : torch.stack([ d['input_ids'] for d in data ]),
        "labels" : torch.stack([ d['input_ids'] for d in data ]),
        "attention_mask" : torch.stack([ mask for (mask, pos_ids) in inv_mask_and_pos_ids ]),
        "position_ids" : torch.stack([ pos_ids for (mask, pos_ids) in inv_mask_and_pos_ids ]),
    }

if __name__ == "__main__":

    ds = datasets.Dataset.load_from_disk("/workspace/tmp/pretrain_ds_first-1M-packed-8192").with_format("torch")

    GLOBAL_RANK = int(os.environ.get('RANK', 0))
    GLOBAL_WORLD_SIZE =int(os.environ.get('WORLD_SIZE', 1))
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    
    if GLOBAL_WORLD_SIZE > 1:
        dist.init_process_group()

    dataloader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=pad_and_collate,
        sampler=DistributedSampler(ds, shuffle=False, drop_last=True) if GLOBAL_WORLD_SIZE > 1 else None,
        num_workers=8
    )

    dliter = iter(dataloader)

    ntokens = 0
    start = time.time()

    for i in tqdm(range(1_000), total=1_000, disable=LOCAL_RANK>0):

        batch = next(dliter)

        if i == 0:
            # Print first batch to double check that data not duplicated across processes
            print(f"{GLOBAL_RANK=}, {batch['input_ids']=}")

        ntokens += GLOBAL_WORLD_SIZE * 8_192

        if LOCAL_RANK == 0 and i % 50 == 0 and i > 0:
            lapsed_time = time.time() - start
            toks_per_sec = ntokens/lapsed_time
            print(f"{ntokens=}, {lapsed_time=:.2f}s, {toks_per_sec=:.0f}")
