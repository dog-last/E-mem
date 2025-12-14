#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-Cached Memory Agent + HotpotQA Dataset Evaluation

Adapted from GAM Framework to use KV-Cached Memory Agent System
"""

import json
import logging
import os
import re
import string
import sys
from collections import Counter

# Add project root to path
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.conversation_manager.factory import create_chat_manager

# ========== Logging Setup ==========

def setup_logger(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger('hotpotqa_eval')
    return logger


# ========== Data Loading ==========

def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA JSON dataset"""
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    data_all = [
        {
            "index": item.get("index", idx),
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": item.get("answers", []),
            "_id": f"hotpotqa-{item.get('index', idx)}"
        }
        for idx, item in enumerate(dataset)
    ]
    
    return data_all


# ========== Context Splitting ==========

def build_context_chunks_for_sample(
    sample: Dict[str, Any], 
    max_tokens: int = 2000, 
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """Split context text into chunks based on token count"""
    if logger is None:
        logger = logging.getLogger('hotpotqa_eval')
        
    context_text = sample.get("context") or ""
    
    if not context_text:
        return []
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    tokens = tokenizer.encode(context_text, disallowed_special=())
    
    if len(tokens) <= max_tokens:
        return [context_text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        
        start_idx = end_idx
    
    return chunks


# ========== Prompt Design ==========

def make_prompt(context: str, question: str) -> str:
    """Create unified prompt (open QA format)"""
    prompt = f"""You are a careful multi-hop reading assistant. 
Use the given Context. 
Answer with ONLY the final answer string; no extra words.

Question:
{question}

Context:
{context}

Answer:
"""
    return prompt


# ========== Answer Evaluation ==========

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def _calculate_f1(pred_answer: str, gold_answers: List[str]) -> float:
    max_f1 = 0.0
    for gold_answer in gold_answers:
        max_f1 = max(max_f1, qa_f1_score(pred_answer, gold_answer))
    return max_f1


# ========== Core Processing Logic ==========

def process_sample(
    sample: Dict[str, Any], 
    sample_index: int, 
    outdir: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    config: dict,
    max_tokens: int = 2000,
    logger: Optional[logging.Logger] = None
):
    """Process a single sample using KV-Cached Memory Agent"""
    if logger is None:
        logger = logging.getLogger('hotpotqa_eval')
        
    sample_id = sample.get("_id", f"sample-{sample_index}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing sample #{sample_index}: {sample_id}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. Build context chunks
        context_chunks = build_context_chunks_for_sample(sample, max_tokens, logger)
        logger.info(f"Number of context chunks: {len(context_chunks)}")
        if context_chunks:
            logger.info(f"First context chunk preview:\n{context_chunks[0][:400]}...")
        
        # Create output directory
        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        logger.info(f"Output directory: {sample_results_dir}")
        
        # 2. Create ChatManager with clean cache
        logger.info("\nStep 1: Create ChatManager")
        storage_mode = config['memory'].get('storage_mode', 'kv_cache')
        agent = create_chat_manager(
            storage_mode=storage_mode,
            model_id=config['model']['model_id'],
            openai_config=config['model']['openai_config'],
            clean_cache_first=True,
            model_context_window=config['model']['model_context_window'],
            attn_implementation=config['model'].get('attn_implementation', 'sdpa'),
            device_map=config['model'].get('device_map', 'auto'),
            router_system_prompt=config['memory'].get('router_system_prompt'),
            quantization_config=config['model'].get('quantization_config'),
            overlap_mode=config['memory'].get('overlap_mode', 'chunk')
        )
        logger.info("[OK] ChatManager created")
        
        # 3. Add context chunks to memory
        logger.info("\nStep 2: Add context chunks to memory")
        for i, context_chunk in enumerate(context_chunks, 1):
            logger.info(f"  Processing context chunk {i}/{len(context_chunks)}...")
            agent.add_memory(context_chunk)
        logger.info(f"[OK] Memory building completed! Added {len(context_chunks)} chunks")
        
        # 4. Query memory and generate answer
        logger.info("\nStep 3: Query memory and generate answer")
        
        question = sample.get("input", "")
        gold_answers = sample.get("answers", [])
        
        logger.info(f"Question: {question}")
        logger.info(f"Standard answers: {gold_answers}")
        
        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "question": question,
            "answers": gold_answers,
            "gold_answers": gold_answers,
            "num_chunks": len(context_chunks)
        }

        try:
            # Query memory to get research summary
            logger.info("Querying memory...")
            research_summary = agent.search_memory(question)
            logger.info("[OK] Memory query completed")
            logger.info(f"Research summary: {research_summary}")
            
            result["research_summary"] = research_summary
            
            # Generate final answer using working model API
            logger.info("Generating final answer...")
            prompt = make_prompt(research_summary, question)
            
            # Call working model API
            from openai import OpenAI
            working_client = OpenAI(api_key=working_api_key, base_url=working_base_url)
            response = working_client.chat.completions.create(
                model=working_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            answer_text = response.choices[0].message.content.strip()
            
            logger.info(f"Model response: {answer_text}")
            
            pred_answer = answer_text
            result["response"] = answer_text
            result["pred"] = pred_answer
            
            # Calculate F1 score
            f1 = _calculate_f1(pred_answer, gold_answers) if pred_answer else 0.0
            result["f1"] = f1
            
            logger.info(f"Predicted answer: {pred_answer}")
            logger.info(f"Standard answers: {gold_answers}")
            logger.info(f"F1 score: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process question: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        # Save result
        results_file = os.path.join(sample_results_dir, "qa_result.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n[OK] Result saved to: {results_file}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Processing completion statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Sample ID: {sample_id}")
        logger.info(f"Number of context chunks: {len(context_chunks)}")
        logger.info(f"Predicted answer: {result.get('pred', 'N/A')}")
        logger.info(f"Standard answers: {gold_answers}")
        logger.info(f"F1 score: {result.get('f1', 0.0):.4f}")
        logger.info(f"Result saved to: {sample_results_dir}")
        
        # Cleanup
        del agent
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Agent cleaned up")
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_index}: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "error": error_msg
        }


# ========== Main Function ==========

def main():
    import argparse
    from datetime import datetime

    import yaml
    
    parser = argparse.ArgumentParser(description="KV-Cached Memory Agent + HotpotQA Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive)")
    
    # Working Generator configuration (for final answer generation)
    parser.add_argument("--working-api-key", type=str, default=None, help="Working model API Key")
    parser.add_argument("--working-base-url", type=str, default=None, help="Working model Base URL")
    parser.add_argument("--working-model", type=str, default=None, help="Working model name")
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        hotpotqa_config = os.path.join(Path(__file__).parent, config_path)
        if os.path.exists(hotpotqa_config):
            config_path = hotpotqa_config
        else:
            config_path = os.path.join(Path(__file__).parent.parent.parent, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override working model config if provided
    working_api_key = args.working_api_key or config['model']['openai_config']['api_key']
    working_base_url = args.working_base_url or config['model']['openai_config']['base_url']
    working_model = args.working_model or config['model']['openai_config']['model']
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"hotpotqa_{timestamp}"
    os.environ['EVAL_SESSION_ID'] = session_id
    
    log_dir = config['logging']['log_dir']
    if not os.path.isabs(log_dir):
        project_root = Path(__file__).parent.parent.parent
        log_dir = os.path.join(project_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"hotpotqa_eval_{timestamp}.log")
    logger = setup_logger(log_file)
    
    logger.info(f"Evaluation session ID: {session_id}")
    logger.info("=" * 60)
    logger.info("KV-Cached Memory Agent + HotpotQA Evaluation")
    logger.info("=" * 60)
    
    # Load data
    dataset_path = config['hotpotqa_eval']['dataset_path']
    if not os.path.isabs(dataset_path):
        project_root = Path(__file__).parent.parent.parent
        dataset_path = os.path.join(project_root, dataset_path)
    
    logger.info(f"Dataset: {dataset_path}")
    all_samples = load_hotpotqa(dataset_path)
    logger.info(f"Total loaded {len(all_samples)} samples")
    
    # Set end index
    if args.end_idx is None:
        args.end_idx = len(all_samples)
    
    # Apply ratio
    ratio = config['hotpotqa_eval'].get('ratio', 1.0)
    if ratio < 1.0:
        args.end_idx = min(args.end_idx, max(1, int(len(all_samples) * ratio)))
    
    logger.info(f"Processing range: {args.start_idx} to {args.end_idx-1} (total {args.end_idx - args.start_idx} samples)")
    
    # Validate range
    if args.start_idx < 0 or args.start_idx >= len(all_samples):
        logger.error(f"Error: Start index {args.start_idx} out of range")
        return
    
    if args.end_idx > len(all_samples):
        logger.warning(f"Warning: End index {args.end_idx} out of range, adjusted to {len(all_samples)}")
        args.end_idx = len(all_samples)
    
    if args.start_idx >= args.end_idx:
        logger.error("Error: Start index must be less than end index")
        return
    
    # Process samples
    outdir = config['hotpotqa_eval']['output_dir']
    if not os.path.isabs(outdir):
        project_root = Path(__file__).parent.parent.parent
        outdir = os.path.join(project_root, outdir)
    
    max_tokens = config['hotpotqa_eval'].get('max_tokens_per_chunk', 2000)
    
    sample_indices = list(range(args.start_idx, args.end_idx))
    logger.info("Starting serial processing of samples...")
    
    all_results = []
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        sample = all_samples[sample_idx]
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting to process sample {sample_idx}/{len(all_samples)-1}")
        logger.info(f"{'='*80}")
        
        try:
            result = process_sample(
                sample, 
                sample_idx, 
                outdir,
                working_api_key,
                working_base_url,
                working_model,
                config,
                max_tokens=max_tokens,
                logger=logger
            )
            logger.info(f"[OK] Sample {sample_idx} processing completed")
            all_results.append(result)
        except Exception as e:
            logger.error(f"[ERROR] Sample {sample_idx} processing failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "sample_id": sample.get("_id", f"sample-{sample_idx}"),
                "error": str(e)
            })
    
    # Calculate statistics
    f1_scores = [r["f1"] for r in all_results if "f1" in r]
    
    if all_results:
        os.makedirs(outdir, exist_ok=True)
        summary_file = os.path.join(outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] Batch results summary saved: {summary_file}")
        
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            total_samples = args.end_idx - args.start_idx
            success_count = len(f1_scores)
            
            statistics = {
                "total_samples": total_samples,
                "success_count": success_count,
                "failed_count": total_samples - success_count,
                "success_rate": success_count / total_samples if total_samples > 0 else 0.0,
                "avg_f1": avg_f1,
                "f1_scores": f1_scores,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx - 1
            }
            
            stats_file = os.path.join(outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            logger.info(f"[OK] Batch test statistics saved: {stats_file}")
            
            logger.info(f"\n{'='*60}")
            logger.info("Batch Test Statistics")
            logger.info(f"{'='*60}")
            logger.info(f"Processed samples: {total_samples}")
            logger.info(f"Successfully answered questions: {success_count}")
            logger.info(f"Failed questions: {total_samples - success_count}")
            logger.info(f"Success rate: {statistics['success_rate']:.2%}")
            logger.info(f"Average F1 score: {avg_f1:.4f}")
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
