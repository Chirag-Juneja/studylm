import re


def stream_parser(stream):
    flag = True
    start_tag = "<think>"
    end_tag = "</think>"
    for token, metadata in stream:
        if metadata["langgraph_node"] == "tools":
            continue
        if token.content == end_tag:
            flag = False
            continue
        if token.content == start_tag:
            flag = True
            continue
        if flag:
            continue
        yield token.content


def remove_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
