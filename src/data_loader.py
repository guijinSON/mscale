import pandas as pd 
from datasets import load_dataset
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

def load_blend_data(language='ko'):
    lan_country_dict = {'ko':'Sounth_Korea','id':'Indonesia','en':"US"}
    mcq = load_dataset("nayeon212/BLEnD",'multiple-choice-questions')['test']
    mcq_filtered = mcq.filter(lambda x: x['country'] == lan_country_dict[language])
     
    question = [i.replace('Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}\n\nA. apple\nB. banana\nC. durian\nD. orange\n\nAnswer:','') for i in mcq_filtered['prompt']]
    prompt = mcq_filtered['prompt']
    A = [eval(i)['A'] for i in mcq_filtered['choices']]
    B = [eval(i)['B'] for i in mcq_filtered['choices']]
    C = [eval(i)['C'] for i in mcq_filtered['choices']]
    D = [eval(i)['D'] for i in mcq_filtered['choices']]
    answer_idx_dict = {'A':0,"B":1,"C":2,"D":3} 
    answer_idx = [answer_idx_dict[x] for x in mcq_filtered['answer_idx']]
    data = {
    'question': question,
    'prompt': prompt,
    'answer': answer_idx,
    'A': A,
    'B': B,
    'C': C,
    'D': D
    }

    df = pd.DataFrame(data)
    return df