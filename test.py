import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence


def test_model(model, tokenizer, PATH):
    start_visualize = []
    end_visualize = []

    with torch.no_grad(), open(f'submissions/{PATH}.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Predicted'])

        rows = []
        # for sample in tqdm(test_dataset, "Testing"):
        for sample in tqdm(indexed_test_dataset, "Testing"):
            input_ids, token_type_ids = [torch.tensor(sample[key], dtype=torch.long, device="cuda") for key in ("input_ids", "token_type_ids")]
            # print(sample)
        
            model.eval()
            with torch.no_grad():
                output = load_model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

            start_logits = output.start_logits
            end_logits = output.end_logits
            start_logits.squeeze_(0), end_logits.squeeze_(0)

            start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
            end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

            probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

            # 토큰 길이 8까지만
            for row in range(len(start_prob) - 8):
                probability[row] = torch.cat((probability[row][:8+row].cpu(), torch.Tensor([0] * (len(start_prob)-(8+row))).cpu()), 0)

            index = torch.argmax(probability).item()

            start = index // len(end_prob)
            end = index % len(end_prob)
            
            # 확률 너무 낮으면 자르기
            if start_prob[start] >= 0 or end_prob[end] >= 0:
                start_str = sample['position'][start][0]
                end_str = sample['position'][end][1]
            else:
                start_str = 0
                end_str = 0

            start_visualize.append((list(start_prob.cpu()), (start, end), (start_str, end_str)))
            end_visualize.append((list(end_prob.cpu()), (start, end), (start_str, end_str)))
            
            rows.append([sample["guid"], sample['context'][start_str:end_str]])

        writer.writerows(rows)
    return writer