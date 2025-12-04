"""Quick prompt testing script for category 2 questions."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
from evaluation.utils import calculate_metrics

# Mock memory result (replace with your actual search result)
MOCK_MEMORY_RESULT = """The memory stored a period of time ago: [1:56 pm on 8 May, 2023] Melanie: Yeah, I painted that lake sunrise last year! It's special to me.  

The exact datetime when Melanie painted the sunrise is not explicitly stated in the context. However, she mentions it was "last year" during her conversation on **2023-05-08 1:56:00**. Since the conversation occurred in May 2023, the painting would have been completed in **2022**, but no specific date is provided for the act of painting itself. The context only references the timing relative to her message.
[Context 72]  
[1:33 pm on 25 August, 2023] Melanie: Yeah, I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative. Have you tried it yet?  

[Context 141]  
[12:09 am on 13 September, 2023] Caroline: Melanie, those bowls are amazing! They each have such cool designs. I love that you chose pottery for your art. Painting and drawing have helped me express my feelings and explore my gender identity. Creating art was really important to me during my transition - it helped me understand and accept myself. I'm so grateful.  

[Context 161]  
[10:31 am on 13 October, 2023] Melanie: Yeah, Here's one I did last week. It's inspired by the sunsets. The colors make me feel calm. What have you been up to lately, artistically?  

[Context 163]  
[10:31 am on 13 October, 2023] Melanie: I painted it because it was calming. I've done an abstract painting too, take a look! I love how art lets us get our emotions out.  

[Context 165]  
[10:31 am on 13 October, 2023] Melanie: I wanted a peaceful blue streaks to show tranquility. Blue calms me, so I wanted the painting to have a serene vibe while still having lots of vibrant colors.  

[Context 176]  
[6:55 pm on 20 October, 2023] Melanie: Thanks, Caroline! It's a great time. Nature and quality time, can't beat it!  

**Note:** There is no explicit mention of Melanie painting a sunrise in the provided context information. The references to art and painting involve sunsets (e.g., Context 141) or abstract works (Context 163), but not specifically a sunrise.

The memory stored just now: The question about when Melanie painted a sunrise is not addressed in any of the provided context information. The conversations between Melanie and Caroline focus on topics such as creating homes for children, self-acceptance, and mutual support, but there is no mention of Melanie painting or any related activity. Therefore, no specific datetime or details about this event are available in the provided context.
"""

# Test question and reference answer
TEST_QUESTION = "When did Melanie paint a sunrise?"
REFERENCE_ANSWER = "2022"

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
    "current_eval": """You are a precise date extraction and normalization assistant.
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
{question}"""
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
