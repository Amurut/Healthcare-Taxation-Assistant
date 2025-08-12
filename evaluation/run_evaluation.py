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

RESULTS_FILE = "evaluation/generated_results_hyde.csv"
SCORES_FILE = "evaluation/ragas_scores_hyde.csv"
SCORES_TEMP_FILE = "evaluation/TEMP_ragas_scores_hyde.csv"

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    try:
        results_df = pd.read_csv(RESULTS_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: `{RESULTS_FILE}` not found. Please run the generation script first.")
        exit()

    print("Columns found in input CSV:", results_df.columns.tolist())
    if 'framework' not in results_df.columns:
        print("Error: The 'framework' column is missing from the input CSV. Please regenerate the results.")
        exit()

    results_df['contexts'] = results_df['contexts'].apply(ast.literal_eval)
    hf_dataset = Dataset.from_pandas(results_df)

    metrics = [
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness,
    ]
    
    print(f"\nRunning RAGAS evaluation on '{RESULTS_FILE}'...")
    result = evaluate(dataset=hf_dataset, metrics=metrics)
    
    evaluation_scores_df = result.to_pandas()
    evaluation_scores_df.to_csv(SCORES_TEMP_FILE, index=False)
    # --- THIS IS THE FIX ---
    # The RAGAS output `result` may drop original metadata.
    # We will combine the original dataframe with the new scores.
    
    # Drop the columns that are in both dataframes from the original df, except the question
    original_metadata_df = results_df.drop(columns=['answer', 'contexts', 'ground_truth'])
    
    # Combine the original metadata with the RAGAS scores
    final_evaluation_df = pd.concat([original_metadata_df, evaluation_scores_df.drop(columns=['user_input'])], axis=1)
    # ----------------------

    print("\n--- Evaluation Complete ---")
    print(final_evaluation_df)
    
    final_evaluation_df.to_csv(SCORES_FILE, index=False)
    print(f"\n✅ Evaluation scores saved to {SCORES_FILE}")

    # Now, groupby will work correctly on the final dataframe
    summary = final_evaluation_df.groupby('framework')[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']].mean()
    print("\n--- Average Scores by Framework ---")
    print(summary)