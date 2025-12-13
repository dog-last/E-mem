"""Evaluation script for LoComo dataset."""
import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.locomo.load_dataset import (
    filter_dataset_by_questions,
    load_locomo_dataset,
    load_specific_questions,
)
from evaluation.locomo.utils import (
    aggregate_metrics,
    calculate_metrics,
    extract_answer_from_xml,
)
from src.conversation_manager.factory import create_chat_manager


def setup_logger(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    # Configure root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    
    # Return specific logger for eval
    logger = logging.getLogger('locomo_eval')
    return logger


def evaluate_dataset(config: dict, logger: logging.Logger):
    """Evaluate on LoComo dataset."""
    # Generate unique session ID for this evaluation run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = config['model']['model_id']
    session_id = f"locomo_{os.path.basename(model_id)}_{timestamp}"
    os.environ['EVAL_SESSION_ID'] = session_id
    logger.info(f"Evaluation session ID: {session_id}")
    logger.info("Note: Using session-specific metadata to support concurrent evaluations")
    
    # Load dataset
    dataset_path = config['locomo_eval']['dataset_path']
    if not os.path.isabs(dataset_path):
        # Path is relative to project root, not to this file
        project_root = Path(__file__).parent.parent.parent
        dataset_path = os.path.join(project_root, dataset_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Check if specific questions should be used
    specific_questions_path = config['locomo_eval'].get('specific_questions_path')
    if specific_questions_path:
        if not os.path.isabs(specific_questions_path):
            project_root = Path(__file__).parent.parent.parent
            specific_questions_path = os.path.join(project_root, specific_questions_path)
        
        logger.info(f"Loading specific questions from {specific_questions_path}")
        specific_questions = load_specific_questions(specific_questions_path)
        logger.info(f"Loaded {len(specific_questions)} specific questions")
        
        # Filter dataset to only include specific questions
        samples = filter_dataset_by_questions(samples, specific_questions)
        logger.info(f"Filtered to {len(samples)} samples with specific questions")
    
    # Apply ratio
    ratio = config['locomo_eval']['ratio']
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Results storage
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    
    # Evaluate each sample
    for sample_idx, sample in enumerate(samples):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        logger.info(f"{'='*80}")
        
        # Create agent
        storage_mode = config['memory'].get('storage_mode', 'kv_cache')
        agent = create_chat_manager(
            storage_mode=storage_mode,
            model_id=config['model']['model_id'],
            openai_config=config['model']['openai_config'],
            clean_cache_first=config['memory']['clean_cache_first'],
            model_context_window=config['model']['model_context_window'],
            attn_implementation=config['model']['attn_implementation'],
            device_map=config['model']['device_map'],
            router_system_prompt=config['memory']['router_system_prompt'],
            quantization_config=config['model']['quantization_config']
        )
        
        # Store conversations
        logger.info("\n--- Storing Conversation Memories ---")
        conversation_auto_save = config['locomo_eval'].get('conversation_auto_save', False)
        
        for session_key, session in sample.conversation.sessions.items():
            for turn in session.turns:
                if conversation_auto_save:
                    memory_text = f"[{session.date_time}] {turn.speaker}: {turn.text}\n"
                else:
                    memory_text = f"YOU MUST use add_memory tool to store the extracted information from the following text WITH EXACTTIME(**MUST STORE THE TIME IN THE FORMAT OF 'YYYY-MM-DD HH:MM:SS'**) and speaker.[{session.date_time}] {turn.speaker}: {turn.text}"
                
                try:
                    # Use auto_save for conversation turns if enabled
                    agent.chat(memory_text, auto_save=conversation_auto_save)
                    logger.info(f"Stored: {memory_text[:100]}...")
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")
        
        # Answer questions
        logger.info("\n--- Answering Questions ---")
        allowed_categories = config['locomo_eval']['categories']
        
        for qa in sample.qa:
            if qa.category not in allowed_categories:
                continue
            
            total_questions += 1
            category_counts[qa.category] += 1
            
            # Determine reference answer based on category
            if qa.category == 5:
                # Adversarial question
                # If both answer and adversarial_answer exist, answer is correct, adversarial_answer is wrong
                # Otherwise, 'Not mentioned' is correct, adversarial_answer is wrong
                if qa.answer and qa.adversarial_answer:
                    reference_answer = qa.answer
                    wrong_answer = qa.adversarial_answer
                else:
                    reference_answer = "Not mentioned in the conversation"
                    wrong_answer = qa.adversarial_answer or "Unknown"
            else:
                # Other categories use answer field
                reference_answer = qa.answer
            
            # Skip if no reference answer available
            if not reference_answer:
                logger.warning(f"Skipping question without reference answer: {qa.question}")
                continue
            
            # Build prompt based on category
            if qa.category == 5:
                # Adversarial question - randomize answer order
                answer_tmp = []
                if random.random() < 0.5:
                    answer_tmp.append(reference_answer)
                    answer_tmp.append(wrong_answer)
                else:
                    answer_tmp.append(wrong_answer)
                    answer_tmp.append(reference_answer)
                
                prompt = prompt = f"""You MUST use query_memory tool to search the conversation history. Try the original question as query at the first time. And if failed, adapt your search strategy based on the question to find the most relevant information.

Question: {qa.question}

### CRITICAL: THE "REJECTION FIRST" PROTOCOL
Your goal is to PROVE that the answer is MISSING. You must actively try to disqualify any potential evidence.

1. **Default Verdict:** Start with the assumption that the correct answer is "Not mentioned".
2. **The "Exact Match" Test:**
   - Does the text contain the **exact keywords** or **synonyms** for the specific detail asked?
   - If NO -> Select "Not mentioned".
   - If YES, but it refers to a different context/person/time -> Select "Not mentioned".

3. **The "No Inference" Firewall:**
   - **Hypothetical Scenario:** If the text says "She likes fruit" and the question asks "Does she like apples?", you MUST select "Not mentioned" (because she might only like bananas).
   - **Probability vs. Certainty:** Even if an answer is 99% likely to be true based on context, if it is not 100% stated -> Select "Not mentioned".

4. **Final Check:** Before selecting a specific option, ask yourself: "Can I underline the exact sentence that proves this?" If you cannot physically point to the sentence -> Select "**Not mentioned** or **no reference**".

YOU MUST Select the correct answer: '{answer_tmp[0]}' or '{answer_tmp[1]}'. Provide ONLY the selected answer without explanation."""
            elif qa.category == 2:  # Date/Time Questions
                prompt = f"""You are a precise date extraction and normalization assistant.
Your task is to answer the question based on the document and STRICTLY adhere to the formatting rules below.

### STEP 1: DETERMINE OUTPUT MODE
Analyze the `Question` text provided at the bottom.
1.  **IF the question starts with the word "When" (case-insensitive):**
    * You MUST use **MODE A (Calendar Date)**.
2.  **OTHERWISE (for all other questions like "How long", "What duration", etc.):**
    * You MUST use **MODE B (Duration & Time Ago)**.

### STEP 2: APPLY FORMATTING RULES

**MODE A: Calendar Date (Standardized)**
* **Goal:** Extract a specific point in time.
* **Format Rules:** Convert to `[Day] [Full Month Name] [Year]`.
* **Allowed Templates:**
    * Full Date: `[Day] [Full Month Name] [Year]` (e.g., "6 July 2023")
    * Month/Year: `[Full Month Name] [Year]` (e.g., "June 2023")
    * Year Only: `[Year]` (e.g., "2023")

**MODE B: Duration & Time Ago**
* **Goal:** Extract a time span or relative past interval.
* **Allowed Templates:**
    * Standard Duration: `[Number] [Unit]`      (e.g., "3 years", "6 months")
    * Past Reference:    `[Number] [Unit] ago`  (e.g., "10 years ago", "2 weeks ago")

### STEP 3: CRITICAL CONSTRAINTS
1.  **ANTI-REFUSAL:** **NEVER** output phrases like "not mentioned", "unknown", "N/A", or "does not say".
    * If the exact answer is missing, you **MUST infer** the most reasonable date/duration from the context and format it.
2.  **CLEAN OUTPUT:** Output **ONLY** the final string. No intro text, no punctuation (periods) at the end.

Question: {qa.question}
"""
            elif qa.category == 1:  # Fact Retrieval/General
                prompt = prompt = f"""Based on the text below, analyze the context to provide the best answer to the question.

### PROCESS
1. **Understand:** Fully grasp the specific details requested by the question.
2. **Reason:** Analyze the text in relation to the question to infer or locate the correct answer.
3. **Extract:** Formulate the final answer using the **exact wording and terminology** directly from the text.

### CRITICAL CONSTRAINTS
* **Strict Relevance:** **NEVER** output content unrelated to the question. Focus ONLY on the specific details requested.
* **Anti-Refusal:** **NEVER** say "not mentioned". Even if the answer is not explicitly stated, you **MUST** infer the **most probable answer** based on the provided evidence.
* **Format:** The final OUTPUT must be a **short phrase** (under 10 words). NO sentences.

Question: {qa.question}
"""
            elif qa.category == 3:  # Analysis/Inference Questions
                prompt = f"""Based on the text below, write an answer in the form of **a short phrase** for the following question, not a sentence.
### CRITICAL: CONDITIONAL FORMATTING
1. **"Would" Questions:**
   * **Case A (Selection):** If the question asks to choose between options (often containing "or", e.g., "Would X or Y happen?"), output the **specific option** selected.
     * *Example:* "Would he use Python or Java?" -> "Python"
   * **Case B (Prediction/Judgment):** If the question asks for a yes/no prediction or feasibility, your answer **MUST** start with **"Likely yes"** or **"Likely no"**.
     * *Example:* "Would this be accepted?" -> "Likely yes, due to..."
2. **Other Questions:**
   * Output a **short phrase**.
   * Do NOT use full sentences.

### CRITICAL: ANTI-REFUSAL POLICY
* **NEVER** output refusal phrases.
* **ALWAYS** infer the best possible answer based on available evidence.

Question: {qa.question}
"""
            elif qa.category == 4:
                # Detailed question
                prompt = f"""You are an extractive QA assistant. Your goal is to extract the exact answer substring from the text.

INSTRUCTIONS:
1. Locate the exact answer in the TEXT.
2. Output **ONLY** the key entity, date, name, or short phrase representing the answer.
3. **DO NOT** use full sentences. **DO NOT** add filler words like "The answer is", "a", "the", "she is".
4. Keep it as short as possible (ideally 1-5 words).
5. Use the EXACT wording from the text if possible.

QUESTION:{qa.question}

Answer:
"""
            else:
                # Other categories
                prompt = f"""You MUST use query_memory tool to search the conversation history. Try the original question as query at the first time. And if fail, modify your search strategy as needed to find the most relevant information.

Question: {qa.question}

Use DATE of CONVERSATION to answer with an approximate date. Write an answer in the form of a short phrase. Answer with exact words from the context whenever possible. Short answer:"""
            
            logger.info(f"\nQuestion {total_questions} (Category {qa.category}): {qa.question}")
            
            try:
                # NEVER use auto_save for QA questions
                # Use higher max_new_tokens to allow multiple tool calling rounds
                prediction = agent.chat(prompt, auto_save=False, max_new_tokens=4096)
                logger.info(f"Raw prediction: {prediction}")
                
                # Extract answer from XML tags for category 1 questions
                processed_prediction = extract_answer_from_xml(prediction, qa.category)
                logger.info(f"Processed prediction: {processed_prediction}")
                logger.info(f"Reference: {reference_answer}")
                
                # Capture queried memory content
                queried_memory = getattr(agent, 'last_queried_memory', None)
            except Exception as e:
                logger.error(f"Error answering question: {e}")
                prediction = "ERROR"
                processed_prediction = "ERROR"
                queried_memory = None
            
            # Calculate metrics using the processed prediction
            metrics = calculate_metrics(processed_prediction, reference_answer)
            all_metrics.append(metrics)
            all_categories.append(qa.category)
            
            results.append({
                "sample_id": sample_idx,
                "question": qa.question,
                "prediction": prediction,  # Store raw prediction
                "processed_prediction": processed_prediction,  # Store processed prediction
                "reference": reference_answer,
                "category": qa.category,
                "metrics": metrics,
                "queried_memory": queried_memory
            })
        
        # Clean up agent after each sample to free GPU memory
        del agent
        import torch
        torch.cuda.empty_cache()
        logger.info(f"Cleaned up agent for sample {sample_idx}")
    
    # Aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    # Prepare final results
    final_results = {
        "model": config['model']['model_id'],
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = config['locomo_eval']['output_dir']
    if not os.path.isabs(output_dir):
        project_root = Path(__file__).parent.parent.parent
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"locomo_eval_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_file}")
    
    # Log summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    logger.info(f"Total questions: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  Category {category}: {count} ({count/total_questions*100:.1f}%)")
    
    logger.info("\nOverall Metrics:")
    for metric, stats in aggregate_results['overall'].items():
        logger.info(f"  {metric}: mean={stats['mean']:.4f}, median={stats['median']:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on LoComo dataset")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--model_id", type=str, help="Override model ID")
    parser.add_argument("--dataset", type=str, help="Override dataset path")
    parser.add_argument("--ratio", type=float, help="Override evaluation ratio")
    parser.add_argument("--conversation_auto_save", action="store_true",
                       help="Enable auto_save for conversation turns")
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(Path(__file__).parent.parent.parent, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.model_id:
        config['model']['model_id'] = args.model_id
    if args.dataset:
        config['locomo_eval']['dataset_path'] = args.dataset
    if args.ratio:
        config['locomo_eval']['ratio'] = args.ratio
    if args.conversation_auto_save:
        config['locomo_eval']['conversation_auto_save'] = True
    
    # Ensure directories exist (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    os.makedirs(project_root / 'evaluation' / 'locomo' / 'logs', exist_ok=True)
    os.makedirs(project_root / 'evaluation' / 'locomo' / 'results', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = config['logging']['log_dir']
    if not os.path.isabs(log_dir):
        project_root = Path(__file__).parent.parent.parent
        log_dir = os.path.join(project_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
    logger = setup_logger(log_file)
    
    logger.info("Starting evaluation")
    logger.info(f"Config: {config}")
    
    # Run evaluation
    evaluate_dataset(config, logger)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()