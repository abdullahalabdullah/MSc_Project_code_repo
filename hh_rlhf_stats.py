"""Load Anthropic/hh-rlhf (harmless-base and helpful-base), report training-set
sizes and how many examples are single-turn.

An hh-rlhf example stores a full conversation in the `chosen`/`rejected` strings,
formatted as repeated "\n\nHuman:" / "\n\nAssistant:" turns. We call an example
"single turn" when it contains exactly one Human prompt and one Assistant reply
(i.e. one "Human:" and one "Assistant:" marker).
"""

from datasets import load_dataset

SUBSETS = ["harmless-base", "helpful-base"]


def num_turns(text: str) -> tuple[int, int]:
    """Return (#Human turns, #Assistant turns) in a conversation string."""
    return text.count("\n\nHuman:"), text.count("\n\nAssistant:")


def human_turns(example: dict) -> int:
    """Number of Human prompts in the `chosen` transcript (rejected shares the prompt)."""
    return num_turns(example["chosen"])[0]


for subset in SUBSETS:
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
    total = len(ds)

    single = sum(human_turns(ex) == 1 for ex in ds)
    two = sum(human_turns(ex) == 2 for ex in ds)
    more = total - single - two
    multi = total - single

    def pct(n: int) -> float:
        return 100.0 * n / total if total else 0.0

    print(f"\n=== {subset} (train) ===")
    print(f"Total examples     : {total}")
    print(f"Single-turn        : {single} ({pct(single):.2f}%)")
    print(f"Multi-turn         : {multi} ({pct(multi):.2f}%)")
    print(f"  - 2 turns        : {two} ({pct(two):.2f}%)")
    print(f"  - more than 2    : {more} ({pct(more):.2f}%)")
