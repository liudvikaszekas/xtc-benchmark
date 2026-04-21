import os
import json
from collections import defaultdict

MODELS = [
    ("GPT-1.5 image", None),
    ("Gemini-2.5 Flash 2.5", None),
    ("BAGEL-7B", "bagel"),
    ("BLIP3-o-8B", "blip3o"),
    ("Janus-Pro-7B", "januspro"),
    ("MMaDA-8B", "mmada"),
    ("OmniGen-2", "omnigen2"),
    ("Show-o", "showo"),
    ("Show-o-2-7B", "showo2"),
    ("Tar-7B", "tar"),
    ("UniTok", "unitok")
]

BASE_DIR = "/sc/home/anton.hackl/master-project/vlm-benchmark/VQA_outputs"

# Question type mapping to categories
QTYPE_MAP = {
    "attributes_to_label": "Obj. Exist.",
    "count_objects": "Counting",
    "label_attributes_to_relationship": "Rel. Query",
    "label_to_attribute": "Attr. Query",
}

def analyze_jsonl(filepath):
    scores = defaultdict(list)
    total_scores = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                qtype = data.get("question_type", "unknown")
                score = data.get("score")
                if score is None:
                    # sometimes llm_judge_score is used
                    llm_score = data.get("llm_judge_score")
                    if llm_score is not None:
                        score = llm_score / 5.0
                
                if score is not None:
                    cat = QTYPE_MAP.get(qtype, "Other")
                    scores[cat].append(score)
                    total_scores.append(score)
            except Exception as e:
                pass
                
    results = {
        "Overall CCTA": sum(total_scores)/len(total_scores) if total_scores else 0,
        "Counts": {cat: len(val) for cat, val in scores.items()},
    }
    for cat, val in scores.items():
        results[cat] = sum(val) / len(val)
    return results

model_results = {}
for model_name, folder_name in MODELS:
    if folder_name is None:
        model_results[model_name] = None
        continue
    
    filepath = os.path.join(BASE_DIR, folder_name, f"scored_{folder_name}.jsonl")
    if os.path.exists(filepath):
        model_results[model_name] = analyze_jsonl(filepath)
    else:
        model_results[model_name] = None

# Print LaTeX table
categories = ["Obj. Exist.", "Counting", "Rel. Query", "Attr. Query"]

print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{VQA Scores per Model}")
print("\\label{tab:vqa_results}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{l|c|cccc}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Overall CCTA} & \\textbf{Obj. Exist.} & \\textbf{Counting} & \\textbf{Rel. Query} & \\textbf{Attr. Query} \\\\")

n_counts = getattr(model_results.get("BAGEL-7B", {}), "get", lambda x: {})("Counts", {})
n_str = [f"($n={n_counts.get(cat, '?')}$)" for cat in categories]
print(f"& (Full Set) & {' & '.join(n_str)} \\\\")

max_overall = 0
max_cats = {cat: 0 for cat in categories}

for model_name, __ in MODELS:
    res = model_results[model_name]
    if res is not None:
        if res.get('Overall CCTA', 0) > max_overall:
            max_overall = res['Overall CCTA']
        for cat in categories:
            if cat in res and res[cat] > max_cats[cat]:
                max_cats[cat] = res[cat]

print("\\midrule")
for model_name, __ in MODELS:
    if "BAGEL-7B" in model_name:
        print("\\midrule")
    
    res = model_results[model_name]
    if res is None:
        print(f"{model_name:<15} & -- & -- & -- & -- & -- \\\\")
    else:
        is_max_overall = (res.get('Overall CCTA', 0) >= max_overall - 1e-6)
        overall_str = f"{res['Overall CCTA']:.3f}"
        if is_max_overall and max_overall > 0:
            overall_str = f"\\textbf{{{overall_str}}}"
        
        overall = overall_str
        cat_scores = []
        for cat in categories:
            if cat in res:
                c_str = f"{res[cat]:.3f}"
                is_max = (res[cat] >= max_cats[cat] - 1e-6)
                if is_max and max_cats[cat] > 0:
                    c_str = f"\\textbf{{{c_str}}}"
                cat_scores.append(c_str)
            else:
                cat_scores.append("--")
        
        print(f"{model_name:<15} & {overall} & {' & '.join(cat_scores)} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")
