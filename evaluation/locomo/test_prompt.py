"""Quick prompt testing script for category 2 questions."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI

from evaluation.utils import calculate_metrics

# Mock memory result (replace with your actual search result)
MOCK_MEMORY_RESULT = """The memory stored a period of time ago: [Context 25]  
[1:14 pm on 25 May, 2023] Melanie: ...My kids are so excited about summer break! We're thinking about going camping next month.  

[Context 167]  
[1:51 pm on 15 July, 2023] Melanie: We even went on another camping trip in the forest.  

[Context 203]  
[8:56 pm on 20 July, 2023] Melanie: We always look forward to our family camping trip.  

**Note:** The specific date for the planned camping trip is not explicitly stated in the context beyond Melanie's mention of "next month" (May 25, 2023). However, the context confirms she mentioned planning to go camping next month on May 25, 2023. Subsequent mentions refer to past or ongoing camping activities but do not specify future dates.
[Context 192]  
[6:55 pm on 20 October, 2023] Melanie: ...we just did it yesterday! The kids loved it and it was a nice way to relax after the road trip.  

[Context 194]  
[6:55 pm on 20 October, 2023] Melanie: ...I love camping trips with my fam...  

[Context 195]  
[6:55 pm on 20 October, 2023] Melanie: ...What do you love most about camping with your fam?  

[Context 196]  
[6:55 pm on 20 October, 2023] Melanie: ...It's a chance to be present and together...  

**Note:** The provided contexts mention Melanie recently completing a camping trip on **20 October 2023**, but there is no explicit information about her planning a future camping trip. The references are retrospective or general statements about her love for camping, not specific future plans.

The memory stored just now: There is no relevant information in the context about Melanie planning to go camping.
"""

# Test question and reference answer
TEST_QUESTION = "When is Melanie planning on going camping?"
REFERENCE_ANSWER = "June 2023"

# OpenAI config
OPENAI_CONFIG = {
    "api_key": "sk-CAD4iMgGyj1cJi8ts6Zz1Essy6H6Ctwl5pIsM3mzPwJvtc1X",
    "base_url": "https://api.chatanywhere.tech/v1"
}

def mock_query_memory(query: str) -> str:
    """Mock the query_memory tool response."""
    return MOCK_MEMORY_RESULT

def test_prompt_with_tool(prompt_template: str, question: str) -> str:
    """Test a prompt simulating the real tool calling scenario."""
    client = OpenAI(**OPENAI_CONFIG)
    
    # System prompt from CHAT_SYS_PROMPT in prompt.py
    system_prompt = """You are a helpful assistant. You will be provided with memory storage.
You must use the memory storage to answer user questions, strictly following the tool usage and output format rules below.

TOOLS USE NOTE:

- Query Memory Tool:
    When you see user input, if it is a question, you must use the Query Memory Tool to answer the question. You must base your answer strictly on the information provided by the Query Memory Tool.
    Whenever you see that you lack sufficient information to answer the question, you must first use this tool to query the memory storage.

- Add Memory Tool:
    You must extract useful information from user input, and then use the Add Memory Tool to add these informations to the memory storage. (If user asks you something instead of providing information, you may not need to store it.)
    
---

STRICT ANSWERING RULES BASED ON CATEGORY:

You will be provided with the category of the question. You MUST adhere to the required output formats when answering!

---

**FALLBACK RULE (Mandatory):**

**If, after diligently using the Query Memory Tool and searching the information provided, you are still unable to deduce the single most correct answer, you must reference the information provided by the Query Memory Tool and provide the most reasonable answer possible. You MUST NOT respond with irrelevant output (e.g., "Not mentioned," "I don't know," etc.)**"""
    
    # Tool definition
    tools = [{
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "This can query some information from memory blocks, so that you can use it to answer user questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query content to be used to query memory."
                    }
                },
                "required": ["query"]
            }
        }
    }]
    
    # User prompt with the actual question prompt template
    user_prompt = f"""Please read the user input carefully and answer the question or follow the instructions to finish the tasks:
    <user_input>{prompt_template.format(question=question)}</user_input>
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # First call - model should call query_memory tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        max_tokens=512
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # If tool was called, provide mock result
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "query_memory":
                # Simulate tool response
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "query_memory",
                    "content": mock_query_memory(tool_call.function.arguments)
                })
        
        # Second call - get final answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=512
        )
        return final_response.choices[0].message.content
    else:
        return response_message.content

# Define different prompt templates to test (from eval_locomo.py category 2)
PROMPTS = {
    "current": """You are a precise date extraction and normalization assistant.
Your task is to answer the question based on the document provided and STRICTLY adhere to the following formatting rules.

### STEP 1: ANALYZE AND SELECT FORMAT
Determine which category the answer falls into and apply the corresponding format.

**CASE A: Specific Calendar Date**
Use this for exact dates mentioned in the text (e.g., "07/06/2023", "July 6th").
-> **Format:** [Specific Day] [Full Month Name] [Year]
-> *Examples:* "6 July 2023", "13 August 2022", "2023"

**CASE B: Relative Date (Time Period/Event Before Date)**
Use this STRICTLY if the text describes a time period or event occurring *before* a known date.
-> **Structure:** the [Time Period] before [Date]
-> **Instruction:** 1. Extract the [Time Period] exactly from the text.
   2. Normalize the [Date] part to match Case A format ([Day] [Month] [Year]).
-> **Format:** the [Time Period] before [Specific Day] [Full Month Name] [Year]
-> *Examples:* - "the Tuesday before 20 July 2023"
   - "the week before 20 July 2023"

**CASE C: Duration**
Use this for time spans.
-> **Format:** [Number] [Unit]
-> *Examples:* "3 years", "6 months"

### STEP 2: GENERATE OUTPUT
- **Logic:** First, select the Case (A, B, or C). Then, generate the string.
- **Constraint:** Output ONLY the final string. NO sentence fragments like "The answer is". NO punctuation at the end.


Question:
{question}""",
"test1":"""
You are a precise date extraction and normalization assistant.
Your task is to answer the question based on the document and STRICTLY adhere to the formatting rules below.

### STEP 1: CATEGORIZE AND NORMALIZE
Determine the category of the answer and apply the corresponding Template.

**CASE A: Specific Calendar Date**
Use this when the answer is a fixed point in time.
* **Rule:** Convert dates to `[Day] [Full Month Name] [Year]` format explicitly.
* **Exception:** If the text *only* provides a Year or Month+Year, output exactly that without making up a day.
* **Templates:**
    * Full Date: `[Day] [Full Month Name] [Year]` (e.g., "6 July 2023")
    * Month/Year: `[Full Month Name] [Year]` (e.g., "June 2023")
    * Year Only: `[Year]` (e.g., "2023")

**CASE B: Relative Date (Strict "Before" Format)**
Use this STRICTLY when the answer is a specific day/week relative to a later date.
* **Structure:** `the [Unit] before [Reference Date]`
* **STRICT CONSTRAINT:** The `[Unit]` slot MUST be exactly one of the **Allowed Units** (week, weekend, day, Monday...Sunday). *Do not use "year", "months", etc.*
* **Action:**
    1.  Extract the **Unit** (e.g., "week", "Tuesday").
    2.  Normalize the **Reference Date** to Format 1 style (`[Day] [Full Month Name] [Year]`).
    3.  Combine strictly.
* **Template:** `the [Unit] before [Day] [Full Month Name] [Year]`
* **Examples:**
    * "The Tuesday prior to 07/20/2023" -> "the Tuesday before 20 July 2023"
    * "a week before July 20th, 2023" -> "the week before 20 July 2023"
    * "the day before Christmas 2023" -> "the day before 25 December 2023"

**CASE C: Duration**
Use this for time spans.
* **Template:** `[Number] [Unit]`
* **Examples:** "3 years", "6 months"

### STEP 2: GENERATE OUTPUT
* Output **ONLY** the final string based on the templates above.
* **NO** trailing punctuation (no periods).
* **NO** intro text (e.g., "The answer is").

Question: {question}
""",
"test3":"""
You are a precise date extraction and normalization assistant.
Your task is to answer the question based on the document and STRICTLY adhere to the formatting rules below.

### CRITICAL INSTRUCTION: STRICT FORMAT ONLY
**You must ONLY output a string that matches Case A, Case B, or Case C.**
**NEVER** output phrases like "not mentioned", "not specified", "unknown", "N/A", "no date found", or "does not say". Even if the answer is difficult to find, you must infer the most likely date or duration and format it.

### STEP 1: CATEGORIZE AND NORMALIZE
Determine the category of the answer and apply the corresponding Template.

**CASE A: Specific Calendar Date**
Use this when the answer is a fixed point in time.
* **Rule:** Convert dates to `[Day] [Full Month Name] [Year]` format explicitly.
* **Exception:** If the text *only* provides a Year or Month+Year, output exactly that without making up a day.
* **Templates:**
    * Full Date: `[Day] [Full Month Name] [Year]` (e.g., "6 July 2023")
    * Month/Year: `[Full Month Name] [Year]` (e.g., "June 2023")
    * Year Only: `[Year]` (e.g., "2023")

**CASE B: Relative Date (Strict "Before" Format)**
Use this STRICTLY when the answer is a specific day/week relative to a later date.
* **Structure:** `the [Unit] before [Reference Date]`
* **STRICT CONSTRAINT:** The `[Unit]` slot MUST be exactly one of the **Allowed Units** (week, weekend, day, Monday...Sunday). *Do not use "year", "months", etc.*
* **Action:**
    1.  Extract the **Unit** (e.g., "week", "Tuesday").
    2.  Normalize the **Reference Date** to Format 1 style (`[Day] [Full Month Name] [Year]`).
    3.  Combine strictly.
* **Template:** `the [Unit] before [Day] [Full Month Name] [Year]`
* **Examples:**
    * "The Tuesday prior to 07/20/2023" -> "the Tuesday before 20 July 2023"
    * "a week before July 20th, 2023" -> "the week before 20 July 2023"
    * "the day before Christmas 2023" -> "the day before 25 December 2023"

**CASE C: Duration**
Use this for time spans.
* **Template:** `[Number] [Unit]`
* **Examples:** "3 years", "6 months"

### STEP 2: GENERATE OUTPUT
* **Constraint 1:** Output **ONLY** the final string.
* **Constraint 2:** **NO** intro text (e.g., "The answer is") and **NO** punctuation (periods) at the end.
* **Constraint 3 (Anti-Refusal):** If you cannot find a perfect match, output the closest possible date or duration entity found in the text. **DO NOT** output "not mentioned".

Question: {question}
"""
}

def main():
    print("="*80)
    print("PROMPT TESTING SCRIPT")
    print("="*80)
    print(f"\nQuestion: {TEST_QUESTION}")
    print(f"Reference: {REFERENCE_ANSWER}")
    print(f"\nContext:\n{MOCK_MEMORY_RESULT[:200]}...\n")
    
    results = []
    for name, prompt in PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        try:
            result = test_prompt_with_tool(prompt, TEST_QUESTION)
            metrics = calculate_metrics(result, REFERENCE_ANSWER)
            
            print(f"Result: {result}")
            print(f"Exact Match: {metrics['exact_match']}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL_f']:.4f}")
            print(f"BLEU-1: {metrics['bleu1']:.4f}")
            
            results.append((name, result, metrics))
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, result, metrics in results:
        print(f"{name:15s} | F1: {metrics['f1']:.4f} | EM: {metrics['exact_match']} | Result: {result[:50]}")
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
