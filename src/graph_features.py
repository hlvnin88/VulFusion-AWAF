import torch


def build_graphcodebert_inputs(tokenizer, code: str, dfg_info: dict, max_len: int = 256):
    """
    Build đầu vào nhánh GraphCodeBERT:
      input_ids, attention_mask, position_idx, dfg_to_code, dfg_to_dfg

    Ghi chú: microsoft/graphcodebert-base khi load qua AutoModel là RoBERTa thuần.
    position_idx truyền vào như position_ids để tương thích interface (paper gọi là
    "structural representation"); graph-attention gốc cần codebase microsoft/CodeBERT.
    """
    code_tokens = tokenizer.tokenize(code)[: max_len - 2]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    pad_id = tokenizer.pad_token_id
    attn_len = len(input_ids)
    if attn_len < max_len:
        input_ids = input_ids + [pad_id] * (max_len - attn_len)

    attention_mask = [1] * attn_len + [0] * (max_len - attn_len)
    # convention GraphCodeBERT: pad=0, CLS=1, code=2..(khác RoBERTa)
    position_idx = [0] + list(range(1, attn_len)) + [0] * (max_len - attn_len)

    dfg = dfg_info.get("dfg", [])
    dfg_to_code, dfg_to_dfg = [], []
    for i, (_var, tok_idx, deps) in enumerate(dfg[:64]):
        cpos = min(tok_idx + 1, attn_len - 1)
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