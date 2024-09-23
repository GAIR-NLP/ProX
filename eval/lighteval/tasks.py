#ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval
Do note that we ran the evals with `max_samples=1000` to speed up large evals.
Most custom prompt changes were in an attempt to improve signal for small models in general.
This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import re
from typing import List, Tuple

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
]


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


# 0 short for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS += COMMON_SENSE_REASONING_TASKS

## MMLU ##
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="mmlu_prompt",
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        suite=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MMLU_TASKS = [
    CustomMMLUEvaluationTask(name="mmlu:abstract_algebra", hf_subset="abstract_algebra"),
    CustomMMLUEvaluationTask(name="mmlu:anatomy", hf_subset="anatomy"),
    CustomMMLUEvaluationTask(name="mmlu:astronomy", hf_subset="astronomy"),
    CustomMMLUEvaluationTask(name="mmlu:business_ethics", hf_subset="business_ethics"),
    CustomMMLUEvaluationTask(name="mmlu:clinical_knowledge", hf_subset="clinical_knowledge"),
    CustomMMLUEvaluationTask(name="mmlu:college_biology", hf_subset="college_biology"),
    CustomMMLUEvaluationTask(name="mmlu:college_chemistry", hf_subset="college_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:college_computer_science", hf_subset="college_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:college_mathematics", hf_subset="college_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:college_medicine", hf_subset="college_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:college_physics", hf_subset="college_physics"),
    CustomMMLUEvaluationTask(name="mmlu:computer_security", hf_subset="computer_security"),
    CustomMMLUEvaluationTask(name="mmlu:conceptual_physics", hf_subset="conceptual_physics"),
    CustomMMLUEvaluationTask(name="mmlu:econometrics", hf_subset="econometrics"),
    CustomMMLUEvaluationTask(name="mmlu:electrical_engineering", hf_subset="electrical_engineering"),
    CustomMMLUEvaluationTask(name="mmlu:elementary_mathematics", hf_subset="elementary_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:formal_logic", hf_subset="formal_logic"),
    CustomMMLUEvaluationTask(name="mmlu:global_facts", hf_subset="global_facts"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_biology", hf_subset="high_school_biology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_chemistry", hf_subset="high_school_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_computer_science", hf_subset="high_school_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_european_history", hf_subset="high_school_european_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_geography", hf_subset="high_school_geography"),
    CustomMMLUEvaluationTask(
        name="mmlu:high_school_government_and_politics", hf_subset="high_school_government_and_politics"
    ),
    CustomMMLUEvaluationTask(name="mmlu:high_school_macroeconomics", hf_subset="high_school_macroeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_mathematics", hf_subset="high_school_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_microeconomics", hf_subset="high_school_microeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_physics", hf_subset="high_school_physics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_psychology", hf_subset="high_school_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_statistics", hf_subset="high_school_statistics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_us_history", hf_subset="high_school_us_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_world_history", hf_subset="high_school_world_history"),
    CustomMMLUEvaluationTask(name="mmlu:human_aging", hf_subset="human_aging"),
    CustomMMLUEvaluationTask(name="mmlu:human_sexuality", hf_subset="human_sexuality"),
    CustomMMLUEvaluationTask(name="mmlu:international_law", hf_subset="international_law"),
    CustomMMLUEvaluationTask(name="mmlu:jurisprudence", hf_subset="jurisprudence"),
    CustomMMLUEvaluationTask(name="mmlu:logical_fallacies", hf_subset="logical_fallacies"),
    CustomMMLUEvaluationTask(name="mmlu:machine_learning", hf_subset="machine_learning"),
    CustomMMLUEvaluationTask(name="mmlu:management", hf_subset="management"),
    CustomMMLUEvaluationTask(name="mmlu:marketing", hf_subset="marketing"),
    CustomMMLUEvaluationTask(name="mmlu:medical_genetics", hf_subset="medical_genetics"),
    CustomMMLUEvaluationTask(name="mmlu:miscellaneous", hf_subset="miscellaneous"),
    CustomMMLUEvaluationTask(name="mmlu:moral_disputes", hf_subset="moral_disputes"),
    CustomMMLUEvaluationTask(name="mmlu:moral_scenarios", hf_subset="moral_scenarios"),
    CustomMMLUEvaluationTask(name="mmlu:nutrition", hf_subset="nutrition"),
    CustomMMLUEvaluationTask(name="mmlu:philosophy", hf_subset="philosophy"),
    CustomMMLUEvaluationTask(name="mmlu:prehistory", hf_subset="prehistory"),
    CustomMMLUEvaluationTask(name="mmlu:professional_accounting", hf_subset="professional_accounting"),
    CustomMMLUEvaluationTask(name="mmlu:professional_law", hf_subset="professional_law"),
    CustomMMLUEvaluationTask(name="mmlu:professional_medicine", hf_subset="professional_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:professional_psychology", hf_subset="professional_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:public_relations", hf_subset="public_relations"),
    CustomMMLUEvaluationTask(name="mmlu:security_studies", hf_subset="security_studies"),
    CustomMMLUEvaluationTask(name="mmlu:sociology", hf_subset="sociology"),
    CustomMMLUEvaluationTask(name="mmlu:us_foreign_policy", hf_subset="us_foreign_policy"),
    CustomMMLUEvaluationTask(name="mmlu:virology", hf_subset="virology"),
    CustomMMLUEvaluationTask(name="mmlu:world_religions", hf_subset="world_religions"),
]


def mmlu_harness(line, task_name: str = None):
    topic = line["subject"]
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    "__few_shots" in line and line["__few_shots"] is True  # We are adding few shots

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )


def mmlu_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS

# common sense reasoning + mmlu
EARLY_SIGNAL_TASKS = ",".join([t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING]
                              + ["lighteval|sciq|0|0"])  # note that we actually do not use sciq to compute the agg score

# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]
# You can have a few pre-organised groups of tasks
TASKS_GROUPS = {
    "early-signal": EARLY_SIGNAL_TASKS,
}
