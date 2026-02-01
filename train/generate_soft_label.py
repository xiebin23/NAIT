import os
os.environ['HF_HOME'] = '/root/autodl-tmp/hf-mirror'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
from tqdm import tqdm


def generate_soft_label(model_name_or_path, dataset, batch_size, tau, stage):
    
    def make_step_rewards(logits, pos_token_mask, candidate_tokens):
        """
        从 logits 中提取每个步骤的奖励分数
        
        Args:
            logits:  模型输出的 logits，shape:  (batch_size, seq_len, vocab_size)
            pos_token_mask: 标记 [POS] token **前一个位置**的 mask，shape:  (batch_size, seq_len)
                            即该位置的输出用于预测 [POS]
            candidate_tokens: [pos_tag_id, neg_tag_id]
        
        Returns:
            list of list: 每个样本的步骤分数列表
        """
        candidate_logits = logits[:, :, candidate_tokens]  # (batch_size, seq_len, 2)
        
        probabilities = F.softmax(candidate_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        pos_probabilities = probabilities[:, :, 0]  # (batch_size, seq_len)
        
        masked_probs = pos_probabilities * pos_token_mask  # (batch_size, seq_len)
        
        all_scores_res = []
        for i in range(masked_probs.size(0)):
            sample_probs = masked_probs[i]  # (seq_len,)
            step_scores = sample_probs[sample_probs != 0].cpu().tolist()
            all_scores_res.append(step_scores)
        
        return all_scores_res

    pos_label_retention = 0
    total_pos_labels = 0
    neg_label_retention = 0
    total_neg_labels = 0

    ds = load_dataset(dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto").eval()

    pos_tag_id = tokenizer.encode('[POS]')[-1]
    neg_tag_id = tokenizer.encode('[NEG]')[-1]
    candidate_tokens = [pos_tag_id, neg_tag_id]

    steps_rewards = []
    for item in tqdm(ds, desc="Calculating PRM scores"):
        steps = item["input"].split("[PRM]")
        prompt_str = "[POS]".join(steps)

        input_ids = tokenizer.encode(
            prompt_str,
            return_tensors="pt"
        ).to(model.device)
        pos_token_mask = torch.zeros_like(input_ids, dtype=torch.float)
        pos_positions = (input_ids == pos_tag_id).nonzero(as_tuple=True)
        for batch_idx, pos in zip(pos_positions[0], pos_positions[1]):
            if pos > 0:
                pos_token_mask[batch_idx, pos - 1] = 1.0
        with torch.no_grad():
            logits = model(input_ids).logits
        step_rewards = make_step_rewards(logits, pos_token_mask, candidate_tokens)[0]

        step_rewards_filtered = []
        for i, step_reward in enumerate(step_rewards):
            if item["value"][i] == "[POS]":
                if abs(step_reward - 1.0) > tau:
                    step_rewards_filtered.append(step_reward)
                else:
                    step_rewards_filtered.append(1.0)
                    pos_label_retention += 1
                total_pos_labels += 1
            elif item["value"][i] == "[NEG]":
                if abs(step_reward - 0.0) > tau:
                    step_rewards_filtered.append(step_reward)
                else:
                    step_rewards_filtered.append(0.0)
                    neg_label_retention += 1
                total_neg_labels += 1
            else:
                raise ValueError(f"Unexpected value: {item['value'][i]}")
        steps_rewards.append(step_rewards_filtered)
        torch.cuda.empty_cache()
    ds = ds.remove_columns('value')
    ds = ds.add_column("value", steps_rewards)

    print(f"POS label retention rate: {pos_label_retention}/{total_pos_labels} = {pos_label_retention/total_pos_labels:.4f}")
    print(f"NEG label retention rate: {neg_label_retention}/{total_neg_labels} = {neg_label_retention/total_neg_labels:.4f}")

    base_model_name = os.path.basename(model_name_or_path)
    base_dataset_name = os.path.basename(dataset)

    ds.save_to_disk(f"./data/{base_dataset_name}-{str(stage)}-sl-{str(tau)}")
    ds.push_to_hub(f"weepcat/{base_dataset_name}-{str(stage)}-sl", base_model_name, split=str(tau))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to load from the datasets library")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing the dataset")
    parser.add_argument("--tau", type=float, default=0.1, help="threshold for filtering step rewards")
    parser.add_argument("--stage", type=int, default=1, help="Stage number for naming the output dataset")
    args = parser.parse_args()
    generate_soft_label(args.model_name_or_path, args.dataset, args.batch_size, args.tau, args.stage)