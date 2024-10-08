import argparse
from vllm import LLM, SamplingParams
from logits import ban_illegal_tokens, get_allowed_token_ids
from src.template import kmmlu_mcqa,BLEND_en_mcqa, indommlu_mcqa, mmlu_mcqa, belebele_mcqa_ko, belebele_mcqa_en, belebele_mcqa_indo
from src.data_loader import load_kmmlu_data, load_blend_data, load_indommlu_data, load_mmlu_data, load_belebele_data
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate language models on KMMLU dataset")
    parser.add_argument("--model", default="facebook/opt-125m", help="Name of the model to evaluate")
    parser.add_argument("--model_version", default="main", help="version of the model to evaluate")
    parser.add_argument("--model_cache_path", default="/", help="path for saving model cache")
    parser.add_argument("--dataset", default="blend", choices=["kmmlu", "indommlu","kmmlu-sample", "indommlu-sample",'blend','mmlu','belebele'], help="Dataset to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--allowed_tokens", nargs='+', default=['A', 'B', 'C', 'D'], help="List of allowed tokens")
    parser.add_argument("--lang", default='none', choices=['en','id','ko'], help="language for benchmark")    
    parser.add_argument("--input_path", default='none', help="input path for the dataset")   
    parser.add_argument("--output_path", default='none', help="path for saving")   
    return parser.parse_args()



def prepare_queries(df,template,dataset_name):
    if dataset_name in ['kmmlu','kmmlu-sample']:
        return [{'query': template.format(row.question, row.A, row.B, row.C, row.D),
                'answer': ['A', 'B', 'C', 'D'][row.answer-1],
                'category': row.Category} for _, row in df.iterrows()]
    
    elif dataset_name in ['blend']:
        return [{'query': template.format(row.question, row.A, row.B, row.C, row.D),
                'answer': ['A', 'B', 'C', 'D'][row.answer]} for _, row in df.iterrows()]
    
    elif dataset_name in ['indommlu','indommlu-sample']:
        return [{'query': template.format(row.question, (eval(row.options) + [""] * (5 - len(eval(row.options))))[0],(eval(row.options) + [""] * (5 - len(eval(row.options))))[1],(eval(row.options) + [""] * (5 - len(eval(row.options))))[2],(eval(row.options) + [""] * (5 - len(eval(row.options))))[3],(eval(row.options) + [""] * (5 - len(eval(row.options))))[4]), 'answer' : row.answer, 'category':row.subject} for _,row in df.iterrows()]
    
    elif dataset_name in ['mmlu']:
        return [{'query': template.format(row.question, *row.choices),
                 'answer' : ['A','B','C','D'][row.answer],
                 'category':row.subject} for _,row in df.iterrows()]
    
    elif dataset_name in ['belebele']:
        return [{'query': template.format(row.question, row.flores_passage,row.mc_answer1,row.mc_answer2,row.mc_answer3,row.mc_answer4),
                 'answer' : ['A','B','C','D'][int(row.correct_answer_num)-1]} for _,row in df.iterrows()]
                    

def evaluate_model(model, queries, sampling_params):
    outputs = model.generate([item['query'] for item in queries], sampling_params)
    for output, item in zip(outputs, queries):
        item['predicted'] = output.outputs[0].text
    return queries

def main():
    args = parse_arguments()

    if args.dataset == "kmmlu":
        df = load_kmmlu_data()
        queries = prepare_queries(df,kmmlu_mcqa,args.dataset)
        
    elif args.dataset == "indommlu":
        df = load_indommlu_data()
        queries = prepare_queries(df,indommlu_mcqa,args.dataset)
        

    elif args.dataset == "kmmlu-sample":
        df = pd.read_csv(args.input_path)
        queries = prepare_queries(df,kmmlu_mcqa,args.dataset)
        
    elif args.dataset == "indommlu-sample":
        df = pd.read_csv(args.input_path)
        queries = prepare_queries(df,indommlu_mcqa,args.dataset)
        
    elif args.dataset == "blend":
        df = load_blend_data(args.lang)
        queries = prepare_queries(df,BLEND_en_mcqa,args.dataset)
    elif args.dataset == "mmlu":
        df = load_mmlu_data()
        queries = prepare_queries(df,mmlu_mcqa,args.dataset)

    elif args.dataset == "belebele":
        df = load_belebele_data(args.lang)
        if args.lang == 'ko':
            queries = prepare_queries(df,belebele_mcqa_ko,args.dataset)
        elif args.lang == 'id':
            queries = prepare_queries(df,belebele_mcqa_indo,args.dataset)
        elif args.lang == 'en':
            queries = prepare_queries(df,belebele_mcqa_en,args.dataset)

    model = LLM(model=args.model,revision = args.model_version, trust_remote_code=True,download_dir = args.model_cache_path)
    allowed_token_ids = get_allowed_token_ids(model, args.allowed_tokens)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1,
        logits_processors=[lambda token_ids, logits: ban_illegal_tokens(token_ids, logits, allowed_token_ids)]
    )

    results = evaluate_model(model, queries, sampling_params)
    output_file = f"{args.output_path}/{args.model.replace('/', '_')}_{args.model_version}_{args.dataset}_{args.lang}.csv"
    pd.DataFrame(results).to_csv(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
