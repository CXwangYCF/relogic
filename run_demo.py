import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import json
import argparse
from utils import fix_seed # Assuming fix_seed is in utils.py
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    local_encoder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                                 "models--sentence-transformers--all-MiniLM-L6-v2")
    revision_file = os.path.join(local_encoder, "refs", "main")
    if os.path.isfile(revision_file):
        with open(revision_file, encoding="utf-8") as f:
            snapshot = os.path.join(local_encoder, "snapshots", f.read().strip())
        if os.path.isfile(os.path.join(snapshot, "modules.json")):
            local_encoder = snapshot
    default_encoder = (local_encoder if os.path.isfile(os.path.join(local_encoder, "modules.json"))
                       else "all-MiniLM-L6-v2")
    parser.add_argument(
        "--task", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters",'pb_math','pb_gsm8k','pb_olympiadbench','pb_omnimath','relogic_logic2text','relogic_spider'],
        help="dataset used for experiment"
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--pred_file", type=str, default="log/multiarith_zero_shot_cot.log",
        help="Path to the JSON/JSONL file containing inference results (e.g., pb_gsm8k_correct.jsonl)."
        # Updated description to reflect new usage
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="demos/multiarith", help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed", type=int, default=192, help="random seed")
    parser.add_argument(
        "--encoder", type=str, default=default_encoder,
        help="sentence-transformer model or local directory; uses bundled weights when available"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    parser.add_argument("--plot", action="store_true",
                        help="Save a cluster plot (requires matplotlib)")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    local_encoder = os.path.isdir(args.encoder)
    print(f"Loading encoder: {args.encoder} (local={local_encoder})")
    encoder = SentenceTransformer(args.encoder, local_files_only=local_encoder)
    task = args.task
    pred_file = args.pred_file
    save_file = args.demo_save_dir
    max_ra_len = args.max_ra_len

    # --- ORIGINAL CLUSTERING AND SAMPLING LOGIC ---
    # Perform kmean clustering
    if task == "last_letters":
        max_ra_len = 7
    if "pb" in task:
        max_ra_len = 10

    if task == "aqua" or task == "last_letters":
        num_clusters = 4
    elif task == "commonsensqa":
        num_clusters = 7
    elif task == "strategyqa":
        num_clusters = 6
    else:
        num_clusters = 8

    # Accept both JSON arrays and inference JSONL files.
    with open(pred_file, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(data, list) or len(data) < num_clusters:
        raise ValueError(f"Need at least {num_clusters} candidate records")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Candidate records must be JSON objects")

    # --- MAPPING from JSON fields to expected variables ---
    # Assuming the JSON contains keys: 'question', 'label', 'generated_critique', 'prediction'
    # Map these to the variables used in the original code
    corpus = []
    questions = []
    rationales = []
    gold_answers = []
    pred_answers = []

    for item in data:
        # Map JSON fields to variables
        q_text = item.get('question', '') # Gets the full prompt string from run_inference
        label = item.get('label', '') # The ground truth answer
        critique = item.get('generated_critique', '') # The full generated critique/rationale
        pred = item.get('prediction', '') # The predicted answer from the model

        # Construct the 'corpus' entry (used for clustering)
        # Original code used: c_question + "\nA:"
        # In pb context, q_text is the full prompt sent to the model (after chat templating or as is)
        # Let's use the original question part if possible, or just the full prompt
        # For pb, q_text might be the full prompt. If so, we might need to reconstruct or store original question separately.
        # For now, let's assume q_text is the part before the model starts generating (or the user's query part).
        # The corpus for clustering should represent the "input" aspect that led to the rationale.
        # Using q_text seems appropriate.
        corpus_entry = q_text + "\nA:" # Follows the original pattern

        corpus.append(corpus_entry)
        questions.append(q_text)
        rationales.append(critique)
        gold_answers.append(str(label)) # Convert to string for consistency
        pred_answers.append(str(pred))  # Convert to string for consistency

    # Verify lengths match
    if not (len(corpus) == len(questions) == len(rationales) == len(gold_answers) == len(pred_answers)):
        print("Warning: Length mismatch between loaded data lists!")
        print(f"Lengths: Corpus={len(corpus)}, Questions={len(questions)}, Rationales={len(rationales)}, Gold={len(gold_answers)}, Pred={len(pred_answers)}")
        # Depending on requirements, you might want to exit here or proceed with minimum length
        min_len = min(len(corpus), len(questions), len(rationales), len(gold_answers), len(pred_answers))
        corpus = corpus[:min_len]
        questions = questions[:min_len]
        rationales = rationales[:min_len]
        gold_answers = gold_answers[:min_len]
        pred_answers = pred_answers[:min_len]

    print(f"Successfully prepared {len(corpus)} data points for clustering.")
    # --- END MAPPING ---

    print(f"Performing clustering into {num_clusters} clusters...")
    corpus_embeddings = encoder.encode(corpus)
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_idx = [[] for i in range(num_clusters)]

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = []
    for i in range(len(clustered_dists)):
        print(f"\n--- Processing Cluster {i+1} ---")
        cluster_indices = clustered_idx[i]

        if not cluster_indices:
            print(f"Cluster {i+1} is empty. Skipping.")
            continue

        # Calculate average lengths for this cluster
        cluster_questions = [questions[idx] for idx in cluster_indices]
        cluster_rationales = [rationales[idx] for idx in cluster_indices]

        avg_q_len = sum(len(q.strip().split()) for q in cluster_questions) / len(cluster_questions)
        avg_ra_lines = sum(len(r.strip().replace("\n\n", "\n").split("\n")) for r in cluster_rationales) / len(cluster_rationales)
        min_q_len = min(len(q.strip().split()) for q in cluster_questions)
        min_ra_len = min(len(r.strip().replace("\n\n", "\n").split("\n")) for r in cluster_rationales)

        print(f"Average question length in Cluster {i+1}: {avg_q_len:.2f}")
        print(f"Minimum question length in Cluster {i+1}: {min_q_len:.2f}")
        print(f"Average rationale lines in Cluster {i+1}: {avg_ra_lines:.2f}")
        print(f"Minimum rationale lines in Cluster {i+1}: {min_ra_len:.2f}")
        
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        if not args.sampling == "center":
            random.shuffle(top_min_dist)

        selected_in_cluster = False
        for element in top_min_dist:
            min_idx = element[0]
            # Get the index within the overall data list
            overall_idx = clustered_idx[i][min_idx]

            # Retrieve mapped data
            c_question = questions[overall_idx]
            c_rationale = rationales[overall_idx].strip()
            c_pred_ans = pred_answers[overall_idx].strip()
            c_gold_ans = gold_answers[overall_idx].strip() if args.debug else None

            current_q_len = len(c_question.strip().split())
            current_ra_lines = len(c_rationale.replace("\n\n", "\n").split("\n"))

            if (current_q_len <= (avg_q_len + min_q_len)/3*2
                and current_ra_lines <= (avg_ra_lines +min_ra_len)/3*2
                and c_pred_ans != ""):
                # Additional check for GSM-like tasks (if applicable)
                # This checks if the prediction appears near the end of the rationale
                if args.task in ["gsm8k", "multiarith", "singleeq", "addsub", "svamp"]:
                    if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                        continue # Skip this sample if prediction is not found at the end

                # Clean up the rationale
                c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                c_rationale = " ".join(c_rationale.split())

                demo_element = {
                    "question": c_question,
                    "rationale": c_rationale,
                    "pred_ans": c_pred_ans,
                    "gold_ans": c_gold_ans, # Will be None if debug is False
                }
                demos.append(demo_element)
                print(f"Selected from Cluster {i+1}, Index {overall_idx}")
                print("Question:", c_question[:100] + "...") # Print first 100 chars for brevity
                print("Rationale (first 150 chars):", c_rationale[:150] + "...")
                print("Pred Ans:", c_pred_ans)
                print("Gold Ans:", c_gold_ans)
                print("")
                selected_in_cluster = True
                break # Break inner loop to go to next cluster
        
        if not selected_in_cluster:
             print(f"No suitable sample found in Cluster {i+1} after filtering.")

    if not demos:
        raise ValueError("No demos passed the original selection filters; use more training candidates")
    os.makedirs(os.path.dirname(os.path.abspath(args.demo_save_dir)), exist_ok=True)
    demos = {"demo": demos}
    print(f"Saving demos to {args.demo_save_dir}...")
    with open(args.demo_save_dir, 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    # Plotting is optional; clustering and demo selection above are unchanged.
    if not args.plot:
        return
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    y_km = clustering_model.fit_predict(corpus_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(corpus_embeddings)
    centers = pca_model.transform(clustering_model.cluster_centers_)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    center_scatter = plt.scatter(centers[:, 0], centers[:, 1], s=250, marker='*', label='centroids', edgecolor='black', c=np.arange(0, num_clusters), cmap=plt.cm.Paired,)
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=[center_scatter], labels=['centroids'])
    plt.title(f'KMeans Clustering Visualization ({num_clusters} clusters)')
    plt.tight_layout()
    plt.savefig(save_file+".png", dpi=600)
    plt.close() # Close the figure to free memory
    print(f"Plot saved to {save_file+'.png'}")

if __name__ == "__main__":
    main()
