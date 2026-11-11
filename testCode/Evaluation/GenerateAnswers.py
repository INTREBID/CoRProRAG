import argparse
import csv
import json
import os
import sys
import traceback
import asyncio
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.App import App
from src.config import settings
from src.prompt import PROMPTS
from src.utils import compute_exact_match, compute_f1


def normalize_method(method: str) -> str:
    method_key = method.strip().lower()
    if method_key in {"corprorag", "logic_graph"}:
        return "corprorag"
    if method_key in {"sgn", "beam_search_kg"}:
        return "SGN"
    raise ValueError(
        "Unsupported method. Available values: corprorag, SGN (or aliases logic_graph, beam_search_kg)."
    )


# ---------------------------- Single Sample Evaluation ----------------------------


def evaluate_traditional_metrics(item):
    """Compute EM, F1, Precision, Recall for a single sample."""
    pred_answer = item["answer"]
    gold_answers = item["real_answer"]

    best_metrics = {"em": 0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Pick best metrics across all gold answers
    for gold in gold_answers:
        em = compute_exact_match(pred_answer, gold)
        f1, precision, recall = compute_f1(pred_answer, gold)

        if em > best_metrics["em"]:
            best_metrics["em"] = em
        if f1 > best_metrics["f1"]:
            best_metrics["f1"] = f1
            best_metrics["precision"] = precision
            best_metrics["recall"] = recall

    return {
        "query": item["query"],
        "prediction": pred_answer,
        "ground_truth": gold_answers,
        "EM": best_metrics["em"],
        "F1": best_metrics["f1"],
        "Precision": best_metrics["precision"],
        "Recall": best_metrics["recall"],
    }


async def evaluate_with_llm(app, item):
    """Use LLM to judge whether the predicted answer is correct."""
    prompt_template = PROMPTS["eval_answer"]
    prompt = prompt_template.format(
        question=item["query"],
        answer=item["answer"],
        gold_answer=item["real_answer"],
    )
    return await app.llm_model_func(prompt=prompt, model="gpt-4o-2024-11-20")


def write_eval_result_to_csv(csv_file, traditional_metrics, llm_result):
    """Append evaluation result to CSV file."""
    row = traditional_metrics.copy()
    if isinstance(row["ground_truth"], list):
        row["ground_truth"] = "|".join(map(str, row["ground_truth"]))
    row["acc"] = llm_result

    csv_line = (
        f'"{row["query"]}","{row["prediction"]}","{row["ground_truth"]}",'
        f'{row["EM"]},{row["F1"]},{row["Precision"]},{row["Recall"]},"{row["acc"]}"\n'
    )

    with open(csv_file, "a", encoding="utf-8") as f:
        f.write(csv_line)


def calculate_acc_ratio(csv_file):
    """Compute the ratio of LLM-judged correct answers in a CSV file."""
    try:
        correct_count = 0
        total_count = 0

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                if row.get("acc", "").strip().lower() == "correct":
                    correct_count += 1

        if total_count == 0:
            print("No data rows in CSV file")
            return 0.0

        acc_ratio = correct_count / total_count
        print("\nACC Results:")
        print(f"Total samples: {total_count}")
        print(f"Correct samples: {correct_count}")
        print(f"ACC (Accuracy): {acc_ratio:.4f} ({acc_ratio * 100:.2f}%)")

        return acc_ratio

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
        return 0.0
    except Exception as e:
        print(f"Error computing ACC: {e}")
        return 0.0


# ---------------------------- Batch Evaluation ----------------------------


def evaluate_json_data(data):
    """Evaluate all samples in JSON data, returning aggregate metrics."""
    total = len(data)
    exact_match = 0
    f1_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    sample_results = []

    for item in data:
        # Extract predicted answer from JSON structure
        if "response" in item["answer"] and "answer" in item["answer"]["response"]:
            pred_answer = item["answer"]["response"]["answer"]
        else:
            pred_answer = item["answer"]["answer"]
        gold_answer = item["real_answer"]

        best_em = 0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

        # Pick best metrics across all gold answers
        for g_answer in gold_answer:
            em = compute_exact_match(pred_answer, g_answer)
            f1, precision, recall = compute_f1(pred_answer, g_answer)

            if em > best_em:
                best_em = em
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall

        exact_match += best_em
        f1_total += best_f1
        precision_total += best_precision
        recall_total += best_recall

        sample_results.append(
            {
                "query": item["query"],
                "prediction": pred_answer,
                "ground_truth": gold_answer,
                "EM": best_em,
                "F1": best_f1,
                "Precision": best_precision,
                "Recall": best_recall,
            }
        )

    # Compute average scores (percentage)
    exact_match = 100.0 * exact_match / total if total else 0.0
    f1 = 100.0 * f1_total / total if total else 0.0
    precision = 100.0 * precision_total / total if total else 0.0
    recall = 100.0 * recall_total / total if total else 0.0

    return {
        "exact_match": exact_match,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "total": total,
        "sample_results": sample_results,
    }


def eval_and_print(input_file):
    """Load data from file, evaluate, and print results."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = evaluate_json_data(data)

    print("Overall Evaluation Results:")
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"Total samples: {results['total']}")

    return results


def write_log_file(log_file, log_data):
    """Write log data to a JSON file."""
    try:
        log_entry = {"timestamp": datetime.now().isoformat(), **log_data}
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=4))
        print(f"Log saved to {log_file}")
    except Exception as e:
        print(f"Error writing log file: {e}")


async def fetch_single_query(app, q_obj, idx, method, max_retries=3, retries=0):
    """Process a single query with retries."""
    try:
        print(f"Processing query {idx + 1}")

        res = await app.query(q_obj["question"], method=method)

        item = {
            "query": q_obj["question"],
            "answer": res["answer"],
            "real_answer": q_obj["answer"],
        }

        eval1 = evaluate_traditional_metrics(item)
        eval2 = await evaluate_with_llm(app, item)

        return {
            "index": idx,
            "query": q_obj["question"],
            "answer_obj": res,
            "real_answer": q_obj["answer"],
            "traditional_metrics": eval1,
            "llm_result": eval2,
        }
    except Exception as e:
        if retries < max_retries:
            print(f"Error at idx={idx}, retry {retries + 1}/{max_retries}: {e}")
            traceback.print_exc()
            return await fetch_single_query(
                app, q_obj, idx, method, max_retries, retries + 1
            )

        print(f"Failed idx={idx} after {max_retries} retries.")
        return {"index": idx, "error": str(e)}


async def process_queries(
    app,
    questions,
    output_json,
    output_csv,
    method="corprorag",
    concurrency=5,
):
    """Process all queries asynchronously with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_fetch(i, q_obj):
        async with semaphore:
            return await fetch_single_query(app, q_obj, i, method)

    tasks = [
        asyncio.create_task(_bounded_fetch(i, question))
        for i, question in enumerate(questions)
    ]

    total = len(tasks)
    outputs = []
    with tqdm(total=total, desc="Processing", unit="query") as pbar:
        for task in asyncio.as_completed(tasks):
            out = await task
            outputs.append(out)
            pbar.update(1)

    final_results = []
    for i, out in enumerate(sorted(outputs, key=lambda x: x.get("index", -1))):
        if out and "error" not in out:
            write_eval_result_to_csv(
                output_csv, out["traditional_metrics"], out["llm_result"]
            )
            final_results.append(
                {
                    "query": out["query"],
                    "answer": out["answer_obj"],
                    "real_answer": out["real_answer"],
                }
            )
        else:
            print(f"Query at index {i} failed: {out.get('error', 'Unknown error')}")

    with open(output_json, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(final_results, ensure_ascii=False, indent=4))
    print(f"Saved {len(final_results)} records to {output_json}")

    return final_results


async def process_once(
    run_id,
    dataset_path,
    eval_count,
    concurrency,
    method="corprorag",
    start_idx=0,
    output_json_file=None,
    output_csv_file=None,
    log_file=None,
):
    """Single run: process questions in parallel and evaluate."""
    print(f"Run {run_id}: Initializing App...")
    app = App(settings)
    print(f"Run {run_id}: App initialized successfully.")

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            questions = json.load(file)

        print(f"Run {run_id}: Loaded {len(questions)} questions from {dataset_path}")

        # Write CSV header
        fieldnames = [
            "query",
            "prediction",
            "ground_truth",
            "EM",
            "F1",
            "Precision",
            "Recall",
            "acc",
        ]
        with open(output_csv_file, "w", encoding="utf-8") as f:
            f.write(",".join(fieldnames) + "\n")

        end_idx = len(questions) if eval_count is None else start_idx + eval_count
        selected_questions = questions[start_idx:end_idx]

        print(
            f"Run {run_id}: Processing {len(selected_questions)} questions (index {start_idx}-{start_idx + len(selected_questions) - 1}) with method '{method}', concurrency={concurrency}..."
        )

        results = await process_queries(
            app=app,
            questions=selected_questions,
            output_json=output_json_file,
            output_csv=output_csv_file,
            method=method,
            concurrency=concurrency,
        )

        print(f"Run {run_id}: All answers generated. Evaluating output files...")
        eval_results = eval_and_print(output_json_file)

        print(f"Run {run_id} ACC results:")
        acc_ratio = calculate_acc_ratio(output_csv_file)

        result_data = {
            "run_id": run_id,
            "status": "success",
            "processed_questions": len(selected_questions),
            "successful_results": len(results),
            "acc_ratio": acc_ratio,
            "exact_match": eval_results["exact_match"],
            "f1": eval_results["f1"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "output_files": {"json": output_json_file, "csv": output_csv_file},
        }

        if log_file:
            write_log_file(log_file, result_data)

        return result_data

    except Exception as e:
        error_msg = f"Run {run_id} failed with error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        error_data = {"run_id": run_id, "status": "error", "error": str(e)}
        if log_file:
            write_log_file(log_file, error_data)

        return error_data

    finally:
        try:
            await app.close_cache_only()
            print(f"Run {run_id}: Cache resources cleaned up.")
        except Exception as e:
            print(f"Run {run_id}: Error during cleanup: {e}")


def build_arg_parser():
    default_dataset = settings.root_dir + "/Input/dataset_for_hipporag2/sample.json"
    parser = argparse.ArgumentParser(
        description="Generate answers and run evaluation."
    )
    parser.add_argument(
        "--method",
        default="corprorag",
        help="Query method: corprorag|SGN",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=1,
        help="Number of questions to evaluate from start index.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent query/evaluation tasks.",
    )
    parser.add_argument(
        "--dataset-path",
        default=default_dataset,
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index in dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.root_dir + "/Output",
        help="Output root directory.",
    )
    return parser


async def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.eval_count is not None and args.eval_count <= 0:
        raise ValueError("--eval-count must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")

    run_id = datetime.now().strftime("%m_%d_%H_%M")
    method = normalize_method(args.method)

    output_json_file = f"{args.output_dir}/Ans/predicted_answers_{run_id}.json"
    output_csv_file = f"{args.output_dir}/Eval/predicted_answers_{run_id}.csv"
    log_file = f"{args.output_dir}/Log/run_log_{run_id}.json"

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    await process_once(
        run_id=run_id,
        dataset_path=args.dataset_path,
        eval_count=args.eval_count,
        concurrency=args.concurrency,
        method=method,
        start_idx=args.start_idx,
        output_json_file=output_json_file,
        output_csv_file=output_csv_file,
        log_file=log_file,
    )


if __name__ == "__main__":
    asyncio.run(main())
