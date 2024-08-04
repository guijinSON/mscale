import pandas as pd 
from datasets import load_dataset

def load_belebele_data(language):
    if language == 'ko':
        return pd.DataFrame(load_dataset("facebook/belebele",split='kor_Hang'))
    elif language == 'id':
        return pd.DataFrame(load_dataset("facebook/belebele",split='ind_Latn'))
    elif language == 'en':
        return pd.DataFrame(load_dataset("facebook/belebele",split='eng_Latn'))
    
def load_mmlu_data():
    stem_fields = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning']
    dfs = [pd.DataFrame(load_dataset("cais/mmlu", field)['test']) for field in stem_fields]
    return pd.concat(dfs)
    
def load_indommlu_data():
    df = pd.DataFrame(load_dataset('indolem/IndoMMLU')['test'])
    df = df[df.group=='STEM']
    df['option_len'] = df.options.apply(lambda x:len(eval(x)))
    df = df[df.option_len==5]
    return df
    
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
    lan_country_dict = {'ko':'South_Korea','id':'Indonesia','en':"US"}
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
