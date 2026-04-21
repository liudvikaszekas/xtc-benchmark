
import pickle
import json
import numpy as np
import argparse
from pathlib import Path
from itertools import islice

def analyze_relationships(pickle_path, meta_path):
    print(f"Loading pickle from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        scene_entries = pickle.load(f)

    print(f"Loading metadata from: {meta_path}")
    with open(meta_path, "r") as f:
        psg_meta = json.load(f)
    
    predicates = psg_meta["predicate_classes"]
    
    print(f"Loaded {len(scene_entries)} scene entries.")
    
    # Stats containers
    winner_scores = []
    runner_up_scores = []
    gaps = []
    no_relation_scores = []
    no_relation_higher_count = 0
    total_pairs = 0
    
    # Top 5 details accumulator
    top_5_stats = {
        "1st": [],
        "2nd": [],
        "3rd": [],
        "4th": [],
        "5th": []
    }

    for entry in scene_entries:
        if "rel_scores" not in entry or "pairs" not in entry:
            continue
            
        for scores in entry["rel_scores"]:
            total_pairs += 1
            
            # scores is an array where index 0 is no_relation, 1..N are predicates
            
            # The logic used in generate_sg.py:
            # best_rel_idx = int(np.argmax(scores[1:])) + 1
            # top_score = float(scores[best_rel_idx])
            
            # Get valid predicate scores (indices 1 to N)
            valid_scores = scores[1:]
            if len(valid_scores) == 0:
                continue
                
            best_valid_idx_local = np.argmax(valid_scores)
            best_rel_idx = best_valid_idx_local + 1
            winner_score = scores[best_rel_idx]
            
            no_rel_score = scores[0]
            
            # Sort valid predicates by score (descending)
            # We want to see the top contenders
            # We carry original indices to know which predicate it is, though strictly for stats we just need scores
            sorted_indices_local = np.argsort(valid_scores)[::-1]
            sorted_scores = valid_scores[sorted_indices_local]
            
            winner_scores.append(winner_score)
            no_relation_scores.append(no_rel_score)
            
            if no_rel_score > winner_score:
                no_relation_higher_count += 1
            
            # Check runner up
            if len(sorted_scores) > 1:
                runner_up = sorted_scores[1]
                runner_up_scores.append(runner_up)
                gaps.append(winner_score - runner_up)
            
            # Collect top 5
            for rank, key in enumerate(["1st", "2nd", "3rd", "4th", "5th"]):
                if rank < len(sorted_scores):
                    top_5_stats[key].append(sorted_scores[rank])
                else:
                    top_5_stats[key].append(0.0) # Or NaN

    print("\n" + "="*50)
    print("RELATIONSHIP SCORE ANALYSIS")
    print("="*50)
    print(f"Total pairs analyzed: {total_pairs}")
    
    if total_pairs == 0:
        print("No pairs found.")
        return

    print("\n--- 'No Relation' vs Winner ---")
    print(f"Avg 'No Relation' score: {np.mean(no_relation_scores):.4f} (std: {np.std(no_relation_scores):.4f})")
    print(f"Avg Winner score:        {np.mean(winner_scores):.4f} (std: {np.std(winner_scores):.4f})")
    print(f"Pairs where 'No Relation' > Winner: {no_relation_higher_count} ({no_relation_higher_count/total_pairs*100:.2f}%)")

    print("\n--- Winner vs Runner-up ---")
    if runner_up_scores:
        print(f"Avg Runner-up score: {np.mean(runner_up_scores):.4f} (std: {np.std(runner_up_scores):.4f})")
        print(f"Avg Gap (1st - 2nd): {np.mean(gaps):.4f} (Median: {np.median(gaps):.4f})")
        
        # Gap distribution
        gaps_arr = np.array(gaps)
        print(f"Gap < 0.05:          {np.sum(gaps_arr < 0.05)} ({np.sum(gaps_arr < 0.05)/len(gaps)*100:.2f}%)")
        print(f"Gap < 0.10:          {np.sum(gaps_arr < 0.10)} ({np.sum(gaps_arr < 0.10)/len(gaps)*100:.2f}%)")
        print(f"Gap > 0.50:          {np.sum(gaps_arr > 0.50)} ({np.sum(gaps_arr > 0.50)/len(gaps)*100:.2f}%)")
    
    print("\n--- Top 5 Scores Statistics ---")
    for key in ["1st", "2nd", "3rd", "4th", "5th"]:
        sc = np.array(top_5_stats[key])
        print(f"{key} Place Avg: {np.mean(sc):.4f} | Median: {np.median(sc):.4f} | Max: {np.max(sc):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", required=True, help="Path to scene-graph.pkl")
    parser.add_argument("--meta", required=True, help="Path to custom_psg.json")
    args = parser.parse_args()
    
    analyze_relationships(args.pickle, args.meta)
