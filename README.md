# mscale
Codebase for multilingual scaling laws.

## Usage

| Dataset | Language   | Command Line                           |
|---------|------------|----------------------------------------|
| MMLU    | English    | python evaluate.py --model "facebook/opt-1.3b" --dataset "mmlu"        |
| MMLU    | Korean     | python evaluate.py --model "facebook/opt-1.3b" --dataset "kmmlu"         |
| MMLU    | Indonesian | python evaluate.py --model "facebook/opt-1.3b" --dataset "indommlu" --allowed_tokens A B C D E     |
| Blend   | English    | python evaluate.py --model "facebook/opt-1.3b" --dataset "blend"  --lang "en"       |
| Blend   | Korean     | python evaluate.py --model "facebook/opt-1.3b" --dataset "blend"  --lang "ko"          |
| Blend   | Indonesian | python evaluate.py --model "facebook/opt-1.3b" --dataset "blend"  --lang "id"      |
| Belebele| English    | python evaluate.py --model "facebook/opt-1.3b" --dataset "belebele" --lang "en"        |
| Belebele| Korean     | python evaluate.py --model "facebook/opt-1.3b" --dataset "belebele" --lang "ko"        |
| Belebele| Indonesian | python evaluate.py --model "facebook/opt-1.3b" --dataset "belebele" --lang "id"        |

