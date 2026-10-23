'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
from utils import *
from local_decoder import LocalDecoder
import os

import re

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def pb_main(dataloader,args,demo,decoder):
    res_data = []
    for i, data in enumerate(dataloader):
        if args.limit_dataset_size > 0 and i >= args.limit_dataset_size:
            break
        d = {}

        x, y = data
        x = x[0]
        y = int(y[0].strip())
        d['question'] = x
        d['label'] = y
        if args.method == "zero_shot":
            x = x
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        elif args.method == "auto_cot":
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        # z = decoder.decode(args, x, max_length)
        z, finish_reason = decoder.decode(x, max_length)

        pred = extract_answer(z)
        try:
            pred = int(pred)
        except:
            pred = None

        d['generated_critique'] = z
        d['prediction'] = pred
        d['match'] = (pred == d['label'])
        d['finish_reason'] = finish_reason
        print(d)
        res_data.append(d)

    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]

    acc1 = np.mean([e['match'] for e in error_data]) * 100
    acc2 = np.mean([e['match'] for e in correct_data]) * 100
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f'{args.dataset} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'{args.dataset}_error.jsonl'), 'w') as f:
        for e in error_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(args.output_dir, f'{args.dataset}_correct.jsonl'), 'w') as f:
        for e in correct_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(args.output_dir, f'{args.dataset}.jsonl'), 'w') as f:
        for e in res_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(args.output_dir, f'{args.dataset}_summary.json'), 'w') as sf:
        json.dump({
            "dataset": args.dataset,
            "error_acc": round(acc1, 4),
            "correct_acc": round(acc2, 4),
            "f1": round(f1, 4)
        }, sf)
    return


import re


def extract_answer_relogic(text: str) -> str:
    """
    Extracts the content following the LAST 'answer ' tag from the generated text.
    """
    # 匹配作为独立单词的 'answer'（前面是单词边界），后面跟空格
    # 这样可以匹配 "answer xxx" 但不会匹配 "The answer is"
    parts = re.split(r'\banswer\s+', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        result = parts[-1].strip()

        # 去掉可能跟在后面的 "The answer is ..."
        result = re.sub(r'\s*The\s+answer\s+is\s+.*$', '', result, flags=re.IGNORECASE).strip()

        # 去掉可能的花括号
        if result.startswith('{') and result.endswith('}'):
            result = result[1:-1].strip()

        return result
    return ""


import json
import os


def relogic_main(dataloader, args, demo, decoder):
    logic_prompt_max = """
                ## Task Description
You are an assistant that translates structured logic expressions into natural language. Based on the following dictionary and rules, convert the input logic expression into fluent and accurate natural language.

---

## Function Dictionary (API → Natural Language Template)

| Function | Natural Language Template | Example |
|----------|---------------------------|---------|
| `count { table }` | "the number of ( {table} )" | count { players } → "the number of ( players )" |
| `only { table }` | "( {table} ) only contains one value" | only { players } → "( players ) only contains one value" |
| `hop { table ; column }` | "( {column} ) of ( {table} )" | hop { players ; height } → "( height ) of ( players )" |
| `avg { table ; column }` | "average value of ( {column} ) of ( {table} )" | avg { players ; height } → "average value of ( height ) of ( players )" |
| `sum { table ; column }` | "sum of ( {column} ) of ( {table} )" | sum { orders ; amount } → "sum of ( amount ) of ( orders )" |
| `max { table ; column }` | "maximum of ( {column} ) of ( {table} )" | max { players ; height } → "maximum of ( height ) of ( players )" |
| `min { table ; column }` | "minimum of ( {column} ) of ( {table} )" | min { players ; height } → "minimum of ( height ) of ( players )" |
| `argmax { table ; column }` | "the row with maximum of ( {column} ) of ( {table} )" | argmax { players ; height } → "the row with maximum of ( height ) of ( players )" |
| `argmin { table ; column }` | "the row with minimum of ( {column} ) of ( {table} )" | argmin { players ; height } → "the row with minimum of ( height ) of ( players )" |
| `nth_max { table ; column ; n }` | "the ( {n} ) maximum value of ( {column} ) of ( {table} )" | nth_max { players ; height ; 3 } → "the ( 3 ) maximum value of ( height ) of ( players )" |
| `nth_min { table ; column ; n }` | "the ( {n} ) minimum value of ( {column} ) of ( {table} )" | nth_min { players ; height ; 2 } → "the ( 2 ) minimum value of ( height ) of ( players )" |
| `nth_argmax { table ; column ; n }` | "the row with the ( {n} ) largest of ( {column} ) of ( {table} )" | nth_argmax { players ; height ; 2 } → "the row with the ( 2 ) largest of ( height ) of ( players )" |
| `nth_argmin { table ; column ; n }` | "the row with the ( {n} ) smallest of ( {column} ) of ( {table} )" | nth_argmin { players ; height ; 2 } → "the row with the ( 2 ) smallest of ( height ) of ( players )" |
| `diff { a ; b }` | "difference between ( {a} ) and ( {b} )" | diff { 10 ; 5 } → "difference between ( 10 ) and ( 5 )" |
| `greater { a ; b }` | "( {a} ) is greater than ( {b} )" | greater { height ; 180 } → "( height ) is greater than ( 180 )" |
| `less { a ; b }` | "( {a} ) is less than ( {b} )" | less { age ; 30 } → "( age ) is less than ( 30 )" |
| `eq { a ; b }` | "( {a} ) is equal to ( {b} )" | eq { grade ; A } → "( grade ) is equal to ( A )" |
| `not_eq { a ; b }` | "( {a} ) is not equal to ( {b} )" | not_eq { city ; Beijing } → "( city ) is not equal to ( Beijing )" |
| `round_eq { a ; b }` | "( {a} ) is about ( {b} )" | round_eq { 3.14 ; 3.1 } → "( 3.14 ) is about ( 3.1 )" |
| `and { a ; b }` | "both ( {a} ) and ( {b} ) are true" | and { greater than 10 ; less than 20 } → "both ( greater than 10 ) and ( less than 20 ) are true" |
| `filter_eq { table ; column ; value }` | "( {column} ) of ( {table} ) that equals ( {value} )" | filter_eq { players ; sport ; basketball } → "( sport ) of ( players ) that equals ( basketball )" |
| `filter_not_eq { table ; column ; value }` | "( {column} ) of ( {table} ) that does not equal ( {value} )" | — |
| `filter_greater { table ; column ; value }` | "( {column} ) of ( {table} ) that is greater than ( {value} )" | — |
| `filter_less { table ; column ; value }` | "( {column} ) of ( {table} ) that is less than ( {value} )" | — |
| `filter_greater_eq { table ; column ; value }` | "( {column} ) of ( {table} ) that is greater than or equal to ( {value} )" | — |
| `filter_less_eq { table ; column ; value }` | "( {column} ) of ( {table} ) that is less than or equal to ( {value} )" | — |
| `all_eq { table ; column ; value }` | "all ( {column} ) of ( {table} ) is ( {value} )" | all_eq { students ; passed ; yes } → "all ( passed ) of ( students ) is ( yes )" |
| `all_not_eq { table ; column ; value }` | "all ( {column} ) of ( {table} ) is not ( {value} )" | — |
| `all_greater { table ; column ; value }` | "all ( {column} ) of ( {table} ) is greater than ( {value} )" | — |
| `all_less { table ; column ; value }` | "all ( {column} ) of ( {table} ) is less than ( {value} )" | — |
| `most_eq { table ; column ; value }` | "most of ( {column} ) of ( {table} ) is ( {value} )" | most_eq { students ; grade ; A } → "most of ( grade ) of ( students ) is ( A )" |
| `most_greater { table ; column ; value }` | "most of ( {column} ) of ( {table} ) is greater than ( {value} )" | — |
| `most_less { table ; column ; value }` | "most of ( {column} ) of ( {table} ) is less than ( {value} )" | — |

---

## Synonym Replacement Rules

To make the translation more natural, the following words can be replaced with synonymous expressions. You may flexibly choose the most natural expression based on context:

### Comparison Synonyms
- **maximum** → can also be expressed as: largest, biggest, greatest, highest, most, top, best, longest, oldest, latest, slowest
- **minimum** → can also be expressed as: smallest, least, lowest, fewest, shortest, worst, youngest, earliest, fastest
- **greater than** → can also be expressed as: larger than, more than, older than, above, over, after, at least, later than, longer than
- **less than** → can also be expressed as: smaller than, fewer than, younger than, below, under, before, shorter than, earlier than
- **most of** → can also be expressed as: the majority of, predominantly

### Number Translation
- Numbers can be expressed as digits (e.g., 3) or words (e.g., three)
- Ordinals (1st, 2nd...) can be translated as: first / the largest, second / the second largest, etc.
- "1st" can also be expressed simply as "the" in superlative contexts (e.g., "the maximum" = "the 1st maximum")

---

## Translation Rules

1. **Nested Expressions**: Start translating from the innermost expression and progressively combine outward.
2. **Filter Series**: First translate the filter condition, then use the filtered table for the outer function.
3. **Arg Series**: These return "rows" — the translation should reflect "the row / the record / the entry".
4. **Pipeline Composition**: When multiple expressions are nested, organize the language according to execution order to maintain logical coherence.
5. **Natural Flow**: The parentheses in the templates are for grouping; in the final output, remove redundant parentheses and adjust wording to produce a fluent, natural sentence.

---

## Examples{"Input Logic Expression": "eq { count { filter_eq { all_rows ; owner ( s ) ; private owner } } ; 5 } = true",
      "answer": "five of the great central railway locomotives and rolling stock were owned by private owners ."}

## Output: The corresponding natural language sentence that conveys the same meaning as the logical form. Your response must end with the tag "answer {your natural language sentence}".
           """
    logic_prompt_lite = """
        Task: Convert the given formal logical expression into a natural language sentence.
        Output: The corresponding natural language sentence that conveys the same meaning as the logical form. Your response must end with the tag "answer {your natural language sentence}".
                Example:1、"input": "eq { count { filter_eq { all_rows ; owner ( s ) ; private owner } } ; 5 } = true",
            "answer": "five of the great central railway locomotives and rolling stock were owned by private owners ."
            2、"input": "and { greater { hop { filter_eq { all_rows ; year ; 1995 } ; pts } ; hop { filter_eq { all_rows ; year ; 1991 } ; pts } } ; and { eq { hop { filter_eq { all_rows ; year ; 1995 } ; pts } ; 13 } ; eq { hop { filter_eq { all_rows ; year ; 1991 } ; pts } ; 1 } } } = true"
            "answer": "mark blundell scored more points in his 1995 formula one race than he did in his 1991 formula one race ."
            3、"input":"and { only { filter_eq { filter_eq { all_rows ; college ; saskatchewan } ; position ; k } } ; eq { hop { filter_eq { filter_eq { all_rows ; college ; saskatchewan } ; position ; k } ; player } ; matt kellett } } = true"
            "answer": "the only kicker drafted by saskatchewan college in the 1998 cfl draft was matt kellett ."
            4、"input":"all_eq { all_rows ; story by ; david simon & ed burns } = true"
            "answer": "all of the wire episodes had the story credited to david simons and ed burns ."
            5、"input":"greater { hop { filter_eq { all_rows ; player ; sura , bob bob sura } ; years for rockets } ; hop { filter_eq { all_rows ; player ; stroud , john john stroud } ; years for rockets } } = true"
            "answer": "bob sura played for the houston rockets later than john stroud ."
    """

    logic_prompt_rule = """"
            Task: Translate the logic expression into natural language.

Requirements:

* Explicit semantic trigger words for every logic operator must be preserved.
* Comparison relations must not be omitted.
* Superlative meanings must not be expressed implicitly.
* Do not replace key operations with pronouns or vague references.
* The original logic should be recoverable through keyword matching from the generated natural language.

Mandatory mappings:

count -> number of
avg -> average
sum -> total / sum
max -> highest / maximum
min -> lowest / minimum
argmax -> row/entity with highest
argmin -> row/entity with lowest
greater -> greater than / more than
less -> less than / lower than
eq -> equal to
not_eq -> not equal to
all -> all
most -> most
only -> only one
diff -> difference between

For nth operators:
The sentence must explicitly contain ordinal words such as:
first / second / third / fourth ...

For filter operators:
The sentence must explicitly contain:
that / whose / with

Forbidden:

* Metaphorical expressions
* Semantic omission
* Replacing key entities with pronouns
* Using the negation word "not" if it does not appear in the logic expression

        Output must contain The corresponding natural language sentence that conveys the same meaning as the logical form. Your response must end with the tag "answer {your natural language sentence}".

            
    """
    logic_prompt = """
            Task: Convert the given formal logical expression into a natural language sentence.
            Output: The corresponding natural language sentence that conveys the same meaning as the logical form. Your response must end with the tag "answer {your natural language sentence}.
            Example:{logic: "eq { count { filter_eq { all_rows ; owner ( s ) ; private owner } } ; 5 } = true",
                "sentence": "five of the great central railway locomotives and rolling stock were owned by private owners."}
        """

    sql_prompt = """
            Task: Generate a natural language question based on the provided SQL query.
            Output: A clear and concise natural language question that the SQL query is designed to answer. Your response must end with the tag "answer {your natural language question}".
            Example:{"SQL": "SELECT avg(LifeExpectancy) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  \"English\" AND T2.IsOfficial  =  \"T\")",
                "sentence": "What is average life expectancy in the countries where English is not the official language?"}
        """

    res_data = []
    BLEC_data = []

    for i, data in enumerate(dataloader):
        if args.limit_dataset_size > 0 and i >= args.limit_dataset_size:
            break
        d = {}
        log_entry = {}  # 新增：用于存储单条日志记录的字典

        x, y = data
        original_input = x[0]  # 保存原始输入
        original_label = y[0]  # 保存原始标签


        prompt = ""
        if args.dataset == "relogic_logic2text":
            prompt = logic_prompt_rule

        elif args.dataset == "relogic_spider":
            prompt = sql_prompt

        x = "Input:" + original_input + "\n" + prompt
        y = original_label
        d['question'] = x
        d['label'] = y
        if args.method == "zero_shot":
            x = x
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        elif args.method == "auto_cot":
            x = demo + x + " " + args.cot_trigger
        else:
            raise ValueError("method is not properly defined ...")
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        time.sleep(0.1)
        z, finish_reason = decoder.decode(x, max_length)

        # pred = extract_answer(z)
        pred = extract_answer_relogic(z)
        # pred = z
        d['generated_critique'] = z
        d['prediction'] = pred
        d['match'] = (pred == d['label'])
        d['finish_reason'] = finish_reason

        log_entry["idx"] = i
        log_entry["logic"] = original_input
        log_entry["pred"] = pred
        log_entry["label"] = d['label']
        BLEC_data.append(log_entry)
        # ------------------------
        print()
        print(d)
        print(i)
        print()

        res_data.append(d)

    os.makedirs(args.output_dir, exist_ok=True)
    # 保存主要的结果文件
    with open(os.path.join(args.output_dir, f'{args.dataset}.jsonl'), 'w') as f:
        for e in res_data:
            f.write(json.dumps(e) + '\n')

    # 创建一个包含日志数据的顶级字典
    final_log_dict = {
        # 文件路径可以根据args或其他参数动态生成，这里用一个示例路径
        f"{args.output_dir}/{args.dataset}_evaluation_log.json": BLEC_data
    }

    log_file_path = os.path.join(args.output_dir, f'{args.dataset}_log_for_evaluation.json')
    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        json.dump(final_log_dict, log_f, ensure_ascii=False, indent=2)

    print(f"Evaluation log saved to: {log_file_path}")
    # ---------------------------

    return


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)

    decoder = LocalDecoder(base_url=args.base_url, model_name=args.model,
                           timeout=args.timeout, temperature=args.temperature,
                           reasoning_effort=None if args.reasoning_effort == "default" else args.reasoning_effort)
    print(f"Local model: {decoder.model_name} at {decoder.base_url}")
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    demo = None
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass

    total = 0
    correct_list = []
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    if 'pb' in args.dataset:
        pb_main(dataloader,args,demo,decoder)
        return

    if 'relogic' in args.dataset:
        relogic_main(dataloader, args, demo, decoder)
        return

    
    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
            # if i < 297:
                continue
            output_line = {}
            
            print('*************************')
            print("{}st data".format(i+1))
                    
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()
            
            # print(x, y)
            
            output_line["question"] = x
            output_line["gold_ans"] = y

            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method == "few_shot_cot":
                x = demo + x
            elif args.method == "auto_cot":
                x = demo + x + " " + args.cot_trigger
            else:
                raise ValueError("method is not properly defined ...")
            
            # Answer experiment by generating text ...
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            # z = decoder.decode(args, x, max_length)
            z,finish_reason = decoder.decode(x, max_length)

            output_line["rationale"] = z

            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct
                # pred = decoder.decode(args, z2, max_length)
                pred,finish_reason = decoder.decode(z2, max_length)
                print(z2 + pred)
            else:
                pred = z
                print(x + pred)

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            
            
            output_line["pred_ans"] = pred
            output_line["wrap_que"] = x
            output_line["finish_reason"] = finish_reason

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')

            # Checking answer ...
            # For relogic datasets (text-to-text), use direct string comparison
            if args.dataset in ("relogic_logic2text", "relogic_spider"):
                # pred is a list after answer_cleansing, get the first element
                pred_str = pred[0] if isinstance(pred, list) else pred
                correct = 1 if pred_str.strip().lower() == y.strip().lower() else 0
            else:
                correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
                #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters",'pb_math','pb_gsm8k','pb_olympiadbench','pb_omnimath','relogic_logic2text','relogic_spider'], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/multiarith", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default=None, help="Local model name; auto-detect if exactly one is available"
    )
    parser.add_argument("--base_url", default=None, help="Local OpenAI-compatible endpoint")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--reasoning_effort", choices=["default", "none", "low", "medium", "high"],
                        default="default", help="Native model thinking setting; default preserves server behavior")
    
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/multiarith", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=None, help="maximum output tokens for reasoning (Logic2Text/Spider: 10240; others: 256)"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=None, help="maximum output tokens for answers (Logic2Text/Spider: 10240; others: 256)"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    parser.add_argument("--dataset_path", default=None, help="Override input JSON (for train candidates or custom data)")
    args = parser.parse_args()
    dataset_override = args.dataset_path
    # Restore the effective limit of the original relogic decoder calls.
    default_max_tokens = 10240 if args.dataset in ("relogic_logic2text", "relogic_spider") else 256
    if args.max_length_cot is None:
        args.max_length_cot = default_max_tokens
    if args.max_length_direct is None:
        args.max_length_direct = default_max_tokens
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "pb_math":
        args.dataset_path = "./dataset/math/math.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "pb_gsm8k":
        args.dataset_path = "./dataset/gsm/gsm.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "pb_olympiadbench":
        args.dataset_path = "./dataset/olympiadbench/olympiadbench.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "pb_omnimath":
        args.dataset_path = "./dataset/omnimath/omnimath.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "relogic_logic2text":
        args.dataset_path = "./dataset/relogic_data/logic2text/raw/test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "relogic_spider":
        args.dataset_path = "./dataset/relogic_data/spider/raw/test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    if dataset_override:
        args.dataset_path = dataset_override

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    
    return args

if __name__ == "__main__":
    main()

