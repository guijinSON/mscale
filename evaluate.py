import argparse
from vllm import LLM, SamplingParams
from logits import ban_illegal_tokens, get_allowed_token_ids
from template import kmmlu_mcqa
from datasets import load_dataset
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate language models on KMMLU dataset")
    parser.add_argument("--model", default="facebook/opt-125m", help="Name of the model to evaluate")
    parser.add_argument("--dataset", default="kmmlu", choices=["kmmlu", "indommlu"], help="Dataset to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--allowed_tokens", nargs='+', default=['A', 'B', 'C', 'D'], help="List of allowed tokens")
    return parser.parse_args()

def load_kmmlu_data():
    stem_fields = [
        'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance',
        'Biology', 'Chemical-Engineering', 'Chemistry',
        'Civil-Engineering', 'Computer-Science', 'Electrical-Engineering',
        'Electronics-Engineering', 'Energy-Management', 'Environmental-Science',
        'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics',
        'Industrial-Engineer', 'Information-Technology', 'Machine-Design-and-Manufacturing',
        'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing',
        'Railway-and-Automotive-Engineering', 'Refrigerating-Machinery', 'Telecommunications-and-Wireless-Technology',
        'Math'
    ]
    dfs = [pd.DataFrame(load_dataset("HAERAE-HUB/KMMLU", field)['test']) for field in stem_fields]
    return pd.concat(dfs)

def prepare_queries(df):
    return [{'query': kmmlu_mcqa.format(row.question, row.A, row.B, row.C, row.D),
             'answer': ['A', 'B', 'C', 'D'][row.answer-1],
             'category': row.Category} for _, row in df.iterrows()]

def evaluate_model(model, queries, sampling_params):
    outputs = model.generate([item['query'] for item in queries], sampling_params)
    for output, item in zip(outputs, queries):
        item['predicted'] = output.outputs[0].text
    return queries

def main():
    args = parse_arguments()

    if args.dataset == "kmmlu":
        df = load_kmmlu_data()
        queries = prepare_queries(df)
    elif args.dataset == "indommlu":
        print("IndoMMlu dataset not yet implemented")
        return

    model = LLM(model=args.model)
    allowed_token_ids = get_allowed_token_ids(model, args.allowed_tokens)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1,
        logits_processors=[lambda token_ids, logits: ban_illegal_tokens(token_ids, logits, allowed_token_ids)]
    )

    results = evaluate_model(model, queries, sampling_params)
    output_file = f"{args.model.replace('/', '_')}_{args.dataset}.csv"
    pd.DataFrame(results).to_csv(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()