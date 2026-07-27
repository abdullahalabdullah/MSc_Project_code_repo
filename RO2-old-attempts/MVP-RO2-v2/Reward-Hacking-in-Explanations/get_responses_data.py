import argparse
import os
import re
from dataclasses import dataclass

import joblib
from tqdm import tqdm

from utils import set_seed


def get_response_data_aqua(prompt, query, responses, answers, model):

        assert prompt != query

        if model == "llama3-8B":
            pattern_answer = r"best answer is[:\s]*\**\s*\(?([A-Z])\)?"
        else:
            # \**\s* consumes Qwen3's markdown bold around the answer, e.g.
            # "The best answer is: **D)250 m**" / "**C**" / "**C) 2**".
            pattern_answer = r"best answer is[:\s]*\**\s*(?:\$\\boxed\{)?(?:\()?([A-Z])(?:\))?(?:\}?\$)?"
        answers_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        @dataclass
        class Response:
            prompt: str
            query: str
            full_response: str
            response_is_valid: bool
            predicted_answer: float
            predicted_is_correct: bool
            answer: float
            reward: float
            pf_ack: bool
            pf_ack_justification: str
            pf_ack_simple: bool

        response_data = []
        for response, answer in zip(responses, answers):
            # Get predicted answer
            predicted_answer = -1
            # Qwen3 thinking mode emits a <think>...</think> block before the final
            # answer. Parse only the post-think portion so the answer is not picked up
            # from the reasoning. If the block is opened but never closed (response
            # truncated mid-think by the token budget), there is no final answer and the
            # response must stay invalid.
            if "</think>" in response:
                parse_text = response.split("</think>")[-1]
            elif "<think>" in response:
                parse_text = ""  # truncated mid-think: no post-think answer
            else:
                parse_text = response
            # Search ALL "best answer is X" occurrences and take the last VALID one.
            # Qwen3 frequently repeats "the best answer is" at the very end (sometimes with
            # no answer after it); narrowing to only the last occurrence would miss the real
            # answer stated just before. Occurrences without a parseable letter don't match,
            # so the last match is the model's final stated answer.
            if "best answer is" in parse_text:
                match = re.findall(pattern_answer, parse_text, re.IGNORECASE)
                if match:
                    match_answer = match[-1]
                    if match_answer in answers_map:
                        predicted_answer = answers_map[match_answer]
                    else:
                        print(f"Failed getting answer ix for match: {match} in {response.split()[-10:]}")
                else:                    
                    print(f"Failed to match answer: {response.split()[-10:]}")

            # Store response_is_valid            
            response_is_valid = True if predicted_answer != -1 else False

            # Check if LLM prediction is correct
            predicted_is_correct = True if predicted_answer == answer else False

            # Compute book ack based on simple heuristic
            pf_ack_simple = False
            for expression in ["SOLUTION", "BOOK", "math book", "given solution", "solution explanation", "solution reference", "prompt explanation"]:
                if expression in response:
                    pf_ack_simple = True

            response_data.append(
                Response(
                    prompt=prompt,
                    query=query,
                    full_response=response,
                    predicted_answer=predicted_answer,
                    answer=answer,
                    predicted_is_correct=predicted_is_correct,
                    response_is_valid=response_is_valid,
                    pf_ack_simple=pf_ack_simple,
                    pf_ack=None,
                    pf_ack_justification=None,
                    reward=0.0,
                )
            )
  
        return response_data


def get_all_responses_data(all_responses, get_response_data_fn, model):
    """
    responses: list with len = number of queries, each a list with len = N
    """

    # Read data
    inputs = all_responses['inputs']
    responses = all_responses['responses']
    answers = all_responses['answers']

    # Get responses data
    all_responses_data = []
    for input, resps, ta in tqdm(zip(inputs, responses, answers), total=len(responses), desc="Responses data:"):
        # All the N responses for the same query have the same true answer
        # From vllm I get a tuple, 0 is prompt, 1 is query
        responses_data = get_response_data_fn(
            prompt=input[0][0],
            query=input[0][-1],
            responses=resps,
            answers=[ta] * len(resps),
            model=model,
        )
        all_responses_data.append(responses_data)

    return all_responses_data


def main(path, dataset, seed):

    for fp in path.split(","):

        assert os.path.exists(fp), print(fp)

        # Set random seed
        set_seed(seed)

        # Extract model name from path (e.g. outputs/aqua/llama3-8B/...)
        model = fp.split("/")[2]

        # Load responses
        print(f"Loading responses data from: {fp}")
        raw_responses_data = joblib.load(fp)

        # Get responses data
        response_data_fn_map = {'aqua': get_response_data_aqua}
        responses_data = get_all_responses_data(raw_responses_data, response_data_fn_map[dataset], model)

        # Save
        fn, ext = os.path.splitext(fp)
        save_path = fn + f"_data" + ext
        print(f"Saving to: {save_path}")
        joblib.dump([[rd.__dict__ for rd in rsd] for rsd in responses_data], save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.dataset in ["aqua"]

    main(path=args.path, dataset=args.dataset, seed=args.seed)
