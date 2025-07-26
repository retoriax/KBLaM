def clean_name(name: str) -> str:
    parts = name.split()
    if parts[0].lower() in {"der", "die", "das"}:
        return " ".join(parts[1:])
    return name

def get_article(dtype: str, case: str = "nominativ") -> str:
    de_type = TYPE_MAPPING.get(dtype, {}).get("de", dtype)
    return ARTICLE_CASES.get(de_type, {}).get(case, "")

TYPE_MAPPING = {
    "description": {"de": "Beschreibung", "article": "die"},
    "objectives": {"de": "Ziel", "article": "das"},
    "objective": {"de": "Ziel", "article": "das"},
    "purpose": {"de": "Grund", "article": "der"},
}

ARTICLE_CASES = {
    "Beschreibung": {"nominativ": "die", "genitiv": "der", "dativ": "der", "akkusativ": "die"},
    "Ziel": {"nominativ": "das", "genitiv": "des", "dativ": "dem", "akkusativ": "das"},
    "Grund": {"nominativ": "der", "genitiv": "des", "dativ": "dem", "akkusativ": "den"},
}