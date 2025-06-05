import argparse
import torch
from datasets import load_dataset , load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--pairs", type=int, default=1)
    return parser.parse_args()


def get_message(instruction, response):

    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
            gating_output=output.gating_output
        return score ,gating_output


def main():

### сделать это красивее
    objectives=[
          'helpsteer-helpfulness',
          'helpsteer-correctness',
          'helpsteer-coherence',
          'helpsteer-complexity',
          'helpsteer-verbosity',
          'ultrafeedback-overall_score',
          'ultrafeedback-instruction_following',
          'ultrafeedback-truthfulness',
          'ultrafeedback-honesty',
          'ultrafeedback-helpfulness',
          'beavertails-is_safe',
          'prometheus-score',
          'argilla-overall_quality',
          'argilla-judge_lm',
          'code-complexity',
          'code-style',
          'code-explanation',
          'code-instruction-following',
          'code-readability'
        ]

    models=[
          "mixtral-8x7b-instruct",
          "mistral-small",
          "mistral-medium",
          "gpt-3.5-turbo-0125",
          "mistral-large",
          "gpt-4-turbo-2024-04-09",
          "claude-3-opus-20240229",
          "claude-3-sonnet-20240229",
          "command-r",
          "command-r-plus",
          "claude-3-haiku-20240307",
          "dbrx-instruct",
          "llama-3-70b-instruct"
        ]

    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')
    #dataset = load_from_disk(args.input_repo) 

    # gather reward
    result = {}
    rm = ArmoRMPipeline(args.reward_model, trust_remote_code=True)

    # gather reward
    for i,model in enumerate(models):
        print(f'gathering reward for {model} response')
        result[f"response_{model}_reward"] = []
        result[f"response_{model}_by_objective"]=[]
        #result["prompt"]=[]
        #result["prompt_category"]=[]
        #result["prompt_uid"]=[]

        
        for row in tqdm(dataset):
            #изменить вот эту функцию 
            reward , gating = rm(get_message(row['prompt'], row[f'{model}_response']))
            gating=gating.squeeze()
            by_objective={objectives[j]:gating[j].item() for j in range(gating.shape[0])}
            result[f"response_{model}_reward"].append(reward)
            result[f"response_{model}_by_objective"].append(by_objective)
            #result["prompt"].append(row["prompt"])
            #result["prompt_category"].append(row["prompt_category"])
            #result["prompt_uid"].append(row["prompt_uid"])
    for k, v in result.items():
        dataset = dataset.add_column(k, v)

    dataset.save_to_disk("rewards")


if __name__ == "__main__":
    main()


#python ./src/ultrafeedback_largebatch/rank.py --input_repo revyu/pulze-intent-v0.1-dataset-unwrapped
