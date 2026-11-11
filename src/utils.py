import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("lightrag")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


async def truncate_list_by_token_size_async(
    list_data: list, key: callable, max_token_size: int
):
    """Async version: Truncate a list of data by token size without blocking event loop"""

    if max_token_size <= 0:
        return []

    # Small lists: process synchronously
    if len(list_data) <= 50:
        return truncate_list_by_token_size(list_data, key, max_token_size)

    # Large lists: use thread pool to avoid blocking event loop
    def _compute_truncation():
        tokens = 0
        for i, data in enumerate(list_data):
            tokens += len(encode_string_by_tiktoken(key(data)))
            if tokens > max_token_size:
                return list_data[:i]
        return list_data

    return await asyncio.to_thread(_compute_truncation)


def truncate_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    truncated_list = []
    for i, data in enumerate(list_data):
        data_tokens = len(encode_string_by_tiktoken(key(data)))
        if tokens + data_tokens > max_token_size:
            truncated_data = key(data)
            truncated_key = encode_string_by_tiktoken(truncated_data)

            remaining_tokens = max_token_size - tokens
            truncated_key = truncated_key[:remaining_tokens]
            truncated_str = decode_tokens_by_tiktoken(truncated_key)
            data_copy = data.copy()
            data_copy["content"] = truncated_str

            truncated_list.append(data_copy)
            return truncated_list
        else:
            tokens += data_tokens
            truncated_list.append(data)
    return truncated_list


async def truncate_by_token_size_async(
    list_data: list, key: callable, max_token_size: int
):
    """Async version: Truncate a list of data by token size without blocking event loop"""

    if max_token_size <= 0:
        return []

    # Small lists: process synchronously
    if len(list_data) <= 50:
        return truncate_by_token_size(list_data, key, max_token_size)

    # Large lists: use thread pool to avoid blocking event loop
    def _compute_truncation():
        tokens = 0
        truncated_list = []
        for i, data in enumerate(list_data):
            data_tokens = len(encode_string_by_tiktoken(key(data)))
            if tokens + data_tokens > max_token_size:
                truncated_data = key(data)
                truncated_key = encode_string_by_tiktoken(truncated_data)

                remaining_tokens = max_token_size - tokens
                truncated_key = truncated_key[:remaining_tokens]
                truncated_str = decode_tokens_by_tiktoken(truncated_key)
                data_copy = data.copy()
                data_copy["content"] = truncated_str

                truncated_list.append(data_copy)
                return truncated_list
            else:
                tokens += data_tokens
                truncated_list.append(data)
        return truncated_list

    return await asyncio.to_thread(_compute_truncation)


def truncate_string_by_token_size(text: str, max_token_size: int) -> str:
    """Truncate a string to fit within the given token limit."""
    if max_token_size <= 0:
        return ""

    tokens = encode_string_by_tiktoken(text)

    if len(tokens) <= max_token_size:
        return text

    truncated_tokens = tokens[:max_token_size]

    truncated_text = decode_tokens_by_tiktoken(truncated_tokens)

    return truncated_text


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
    )


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def fix_json_format(json_string):
    json_string = json_string.strip()
    # Remove trailing commas
    json_string = re.sub(r",\s*(\]|\})", r"\1", json_string)
    # Add missing commas between adjacent brackets
    json_string = re.sub(r"(\}|\])\s*(\{|\[)", r"\1,\2", json_string)
    # Fix bracket matching
    open_brackets = {"[": "]", "{": "}"}
    close_brackets = {"]": "[", "}": "{"}
    stack = []
    result = list(json_string)
    for i, char in enumerate(json_string):
        if char in open_brackets:
            stack.append((char, i))
        elif char in close_brackets:
            if stack and stack[-1][0] == close_brackets[char]:
                stack.pop()
            else:
                # Remove unmatched closing bracket
                result[i] = ""
    # Append missing opening brackets
    while stack:
        _, index = stack.pop()
        result.insert(index, open_brackets[result[index]])
    return "".join(result)


def escape_braces(s):
    return s.replace("{", "{{").replace("}", "}}")

def clean_invalid_escapes(json_str):
    """Remove invalid backslash escapes from JSON strings."""
    valid_escape_pattern = r'\\(["\\/bfnrt])'
    all_backslashes_pattern = r"\\(.)"

    def replace_invalid(match):
        if re.match(valid_escape_pattern, match.group(0)):
            return match.group(0)
        return match.group(1)

    return re.sub(all_backslashes_pattern, replace_invalid, json_str)


async def parse_json_response(response: str, use_model_func):
    """Parse JSON response, using LLM to fix on failure."""
    from .prompt import PROMPTS  # avoid circular import

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        print(f"Error occurred during JSON decoding: {e}\nContent:\n{response}")
        retry_prompt = PROMPTS["fix_json"] + response
        retry_response = await use_model_func(prompt=retry_prompt)
        result = json.loads(retry_response.strip())
        return result


def normalize_list(list_objs, key="weighted_score"):
    """Normalize values of the given key across list objects."""
    if not list_objs:
        return
    max_sc = max(o[key] for o in list_objs) or 1.0
    for o in list_objs:
        o[key] /= max_sc


def split_chunk_ids(source_id: str) -> list:
    """Split source_id string into a list of chunk IDs."""
    return source_id.split("<SEP>") if source_id else []


# ============================================================================
# Evaluation Utilities
# ============================================================================


def normalize_answer(s):
    """Normalize answer: lowercase, remove articles, punctuation, and extra whitespace."""
    import string

    def remove_articles(text):
        if isinstance(text, str):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        return text

    def white_space_fix(text):
        if isinstance(text, str):
            return " ".join(text.split())
        return text

    def remove_punc(text):
        if isinstance(text, str):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        return text

    def lower(text):
        if isinstance(text, str):
            return text.lower()
        return text

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """Tokenize a normalized string into a word list."""
    if not s:
        return []
    if isinstance(s, int):
        s = str(s)
    return normalize_answer(s).split()


def compute_exact_match(a_pred, a_gold):
    """Compute exact match score between predicted and gold answers."""
    return int(normalize_answer(a_pred) == normalize_answer(a_gold))


def compute_f1(a_pred, a_gold):
    """Compute F1 score, precision, and recall between predicted and gold answers."""
    import collections

    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        if len(gold_toks) == 0 and len(pred_toks) == 0:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall
