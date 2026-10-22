import torch


def build_graphcodebert_inputs(tokenizer, code: str, dfg_info: dict, max_len: int = 256):
    """
    Build GraphCodeBERT-like inputs:
    - input_ids
    - attention_mask
    - position_idx
    - dfg_to_code
    - dfg_to_dfg

    This is a practical re-implementation interface for training.
    """
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: max_len - 2]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # padding
    pad_id = tokenizer.pad_token_id
    attn_len = len(input_ids)
    if attn_len < max_len:
        input_ids = input_ids + [pad_id] * (max_len - attn_len)

    attention_mask = [1] * attn_len + [0] * (max_len - attn_len)

    # position idx: GraphCodeBERT convention (non-pad positive)
    position_idx = list(range(1, attn_len + 1)) + [0] * (max_len - attn_len)

    # map DFG nodes to code token positions (heuristic by token index)
    dfg = dfg_info.get("dfg", [])
    dfg_to_code = []
    dfg_to_dfg = []

    code_len = attn_len
    for i, (_var, tok_idx, deps) in enumerate(dfg[:64]):  # cap dfg nodes
        cpos = min(tok_idx + 1, code_len - 1)  # +1 for CLS shift
        dfg_to_code.append((i, cpos))
        for d in deps:
            if d < 64:
                dfg_to_dfg.append((i, d))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "position_idx": torch.tensor(position_idx, dtype=torch.long),
        "dfg_to_code": dfg_to_code,
        "dfg_to_dfg": dfg_to_dfg,
    }