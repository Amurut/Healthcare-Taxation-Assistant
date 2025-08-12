# evaluation/run_evaluation_multi_query.py
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
import os
import ast

# --- CONFIGURATION ---
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY must be set for RAGAS evaluation.")

RESULTS_FILE = "evaluation/generated_results_multi_query.csv"
SCORES_FILE = "evaluation/ragas_scores_multi_query.csv"

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    try:
        results_df = pd.read_csv(RESULTS_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: `{RESULTS_FILE}` not found. Please run the generation script first.")
        exit()

    results_df['contexts'] = results_df['contexts'].apply(ast.literal_eval)
    hf_dataset = Dataset.from_pandas(results_df)

    metrics = [
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness,
    ]
    
    print(f"Running RAGAS evaluation on '{RESULTS_FILE}'...")
    result = evaluate(dataset=hf_dataset, metrics=metrics)
    
    evaluation_df = result.to_pandas()
    
    original_metadata_df = results_df.drop(columns=['answer', 'contexts', 'ground_truth'])
    final_evaluation_df = pd.concat([original_metadata_df, evaluation_df], axis=1)

    print("\n--- Evaluation Complete ---")
    print(final_evaluation_df)
    
    final_evaluation_df.to_csv(SCORES_FILE, index=False)
    print(f"\n✅ Evaluation scores saved to {SCORES_FILE}")

    summary = final_evaluation_df.groupby('framework')[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']].mean()
    print("\n--- Average Scores by Framework ---")
    print(summary)