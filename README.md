# mscale
Codebase for multilingual scaling laws.

## Usage
Using default settings
```python
python script_name.py --model "facebook/opt-1.3b" --dataset "kmmlu"
```

Specify decoding configurations

```python
python script_name.py --model "facebook/opt-1.3b" --dataset "kmmlu" --temperature 0.7 --top_p 0.9
```

Specify allowed tokens

```python
python script_name.py --model "facebook/opt-1.3b" --dataset "kmmlu" --temperature 0.7 --top_p 0.9 --allowed_tokens A B C D E
```
