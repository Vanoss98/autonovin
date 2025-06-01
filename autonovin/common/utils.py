import re


def link_remover(md_text: str) -> str:
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    md_text = re.sub(r'\[.*?\]\(.*?\)', '', md_text)
    return md_text
