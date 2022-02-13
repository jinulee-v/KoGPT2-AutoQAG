import torch
import torch.nn.functional as F


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k):
    input_ids = [vocab[tok.bos_token]] + tok(sent)["input_ids"] + [vocab["<unused0>"]]
    count = 0
    generated_text = ''

    if len(input_ids) > 1024:
        return 0

    while 1:
        predicts = model(torch.tensor(input_ids).unsqueeze(0).to(next(model.parameters()).device))
        pred = predicts[0]

        # temper
        logits = pred
        logits = logits[:, -1, :] / temperature
        # top k
        logits = top_k_logits(logits, top_k)
        # top p
        logits = top_p_logits(logits, top_p=top_p)

        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1).item()

        input_ids.append(prev)

        if prev == tok.convert_tokens_to_ids("</s>") or count > text_size:
            sent = tok.convert_ids_to_tokens(input_ids)
            sent = "".join(sent).replace('▁', ' ')
            return sent

        count += 1