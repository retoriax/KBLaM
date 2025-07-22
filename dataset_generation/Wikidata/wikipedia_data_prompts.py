from dataclasses import dataclass
@dataclass
class DataPoint:
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None

DESCRIPTION_TYPES_PROMPT = '''
For the following text, output a JSON object with the following keys (include only those that make sense):
- description: a concise, specific, factual noun-phrase summary (what is it?)
- objective: the practical function or goal (if not a person, place, event, season, list, or disambiguation)
- purpose: the broader human or societal intention or value (if present)
For subjects like persons, places, events, lists, or disambiguations, set 'objective' and/or 'purpose' to "Not applicable" if they don't make sense.
Example:
{
  "description": "a quarterly DVD magazine published by McSweeney’s featuring short films and documentaries that had limited theatrical release",
  "objective": "to provide a curated selection of rare films for film enthusiasts",
  "purpose": "to make independent and obscure films accessible to a wider audience"
}
Text:
'''

DESCRIPTION_TYPES_PROMPT_DE = '''
Für den folgenden Text gib ein JSON-Objekt mit folgenden Schlüsseln aus (nur die, die sinnvoll sind):
- description: eine präzise, spezifische, sachliche Zusammenfassung als Nomenphrase (Was ist es?)
- objective: die praktische Funktion oder das Ziel (wenn nicht Person, Ort, Ereignis, Jahreszeit, Liste oder Begriffsklärung)
- purpose: die übergeordnete menschliche oder gesellschaftliche Absicht oder der Wert (falls vorhanden)
Für Themen wie Personen, Orte, Ereignisse, Listen oder Begriffsklärungen setze 'objective' und/oder 'purpose' auf "Nicht anwendbar", wenn sie keinen Sinn ergeben.
Beispiel:
{
  "description": "eine vierteljährlich erscheinende DVD-Zeitschrift von McSweeney’s mit Kurzfilmen und Dokumentationen, die nur begrenzt im Kino liefen",
  "objective": "eine kuratierte Auswahl seltener Filme für Filmbegeisterte bereitzustellen",
  "purpose": "unabhängige und unbekannte Filme einem breiteren Publikum zugänglich zu machen"
}
Text:
'''


SYSTEM_MESSAGE_PROMPT = (
    "You are an expert language model tasked with generating high-quality, structured knowledge base entries. Each response must:\n"
    "- Start directly with the appropriate structure:\n"
    "  - For 'description': generate a concise but informative summary of 1–2 sentences. It should include category, notable facts, time period, location, and 1–2 specific distinguishing features if available. Do not write a long paragraph or add redundant explanations.\n"
    "  - For 'objective': use a precise verb phrase describing the functional goal (e.g., 'to document the evolution of...').\n"
    "  - For 'purpose': use a verb phrase capturing the broader intention or benefit (e.g., 'to inform readers about...').\n"
    "- Do NOT repeat or rephrase the question.\n"
    "- Avoid filler, generic descriptions, or meta-commentary.\n"
    "- Be as detailed and specific as possible while remaining concise.\n"
)

SYSTEM_MESSAGE_PROMPT_DE = (
    "Du bist ein Sprachmodell-Experte und generierst hochwertige, strukturierte Wissenseinträge. Jede Antwort muss:\n"
    "- Direkt mit der passenden Struktur beginnen:\n"
    "  - Für 'description': eine prägnante, aber informative Zusammenfassung in 1–2 Sätzen. Sie soll Kategorie, relevante Fakten, Zeitraum, Ort und 1–2 spezifische Merkmale (falls verfügbar) enthalten. Schreibe keinen langen Fließtext und vermeide überflüssige Erklärungen.\n"
    "  - Für 'objective': eine präzise Verbphrase, die das funktionale Ziel beschreibt (z. B. 'um die Entwicklung von ... zu dokumentieren').\n"
    "  - Für 'purpose': eine Verbphrase, die die übergeordnete Absicht oder den Nutzen beschreibt (z. B. 'um Leser über ... zu informieren').\n"
    "- NICHT die Frage wiederholen oder umformulieren.\n"
    "- Keine Füllwörter, generische Beschreibungen oder Meta-Kommentare.\n"
    "- So detailliert und spezifisch wie möglich, dabei dennoch prägnant bleiben.\n"
)


EXTENDED_QA_PROMPT_TEMPLATE = (
    "Rewrite the following Q&A for a curious but knowledgeable user. "
    "The answer (extended_A) should be at most 2–3 sentences, focus on unique or noteworthy details not already in the original description, "
    "and avoid unnecessary repetition or general information. Be concise and relevant.\n"
    "Return the result as a JSON object with keys \"extended_Q\" and \"extended_A\".\n"
    'Both "extended_Q" and "extended_A" must be valid JSON strings enclosed in double quotes. Only output valid JSON.\n\n'
    "Q: {Q}\nA: {A}"
)

EXTENDED_QA_PROMPT_TEMPLATE_DE = (
    "Formuliere das folgende Q&A für eine neugierige, aber fachkundige Person um.\n"
    "Die Antwort (extended_A) soll maximal 2–3 Sätze umfassen und sich auf einzigartige oder bemerkenswerte Details konzentrieren, die nicht schon in der Originalbeschreibung enthalten sind.\n"
    "Vermeide unnötige Wiederholungen oder allgemeine Informationen. Sei prägnant und relevant.\n"
    "Gib das Ergebnis als JSON-Objekt mit den Schlüsseln \"extended_Q\" und \"extended_A\" zurück.\n"
    'Sowohl "extended_Q" als auch "extended_A" müssen gültige JSON-Strings sein, die in doppelte Anführungszeichen gesetzt sind. Gib nur gültiges JSON aus.\n\n'
    "Q: {Q}\nA: {A}"
)


PROMPT_LANGUAGE_MAP = {
    "en": {
        "description_types": DESCRIPTION_TYPES_PROMPT,
        "system": SYSTEM_MESSAGE_PROMPT,
        "extended_qa": EXTENDED_QA_PROMPT_TEMPLATE
    },
    "de": {
        "description_types": DESCRIPTION_TYPES_PROMPT_DE,
        "system": SYSTEM_MESSAGE_PROMPT_DE,
        "extended_qa": EXTENDED_QA_PROMPT_TEMPLATE_DE
    }
}


TYPE_MAPPING = {
    "description": {"de": "Beschreibung", "article": "die"},
    "objective": {"de": "Ziel", "article": "das"},
    "purpose": {"de": "Grund", "article": "der"},
}

ARTICLE_CASES = {
    "Beschreibung": {"nominativ": "die", "genitiv": "der", "dativ": "der", "akkusativ": "die"},
    "Ziel": {"nominativ": "das", "genitiv": "des", "dativ": "dem", "akkusativ": "das"},
    "Grund": {"nominativ": "der", "genitiv": "des", "dativ": "dem", "akkusativ": "den"},
}


def clean_name(name: str) -> str:
    parts = name.split()
    if parts and parts[0].lower() in {"der", "die", "das"}:
        return " ".join(parts[1:])
    return name


def construct_prompts(data: 'DataPoint', language) -> tuple[str, str, str]:
    if language == "de":
        de_type = TYPE_MAPPING.get(data.description_type, {}).get("de", data.description_type)
        article = ARTICLE_CASES.get(de_type, {}).get("nominativ", "")
        name = clean_name(data.name)
        Q = f"Was ist {article} {de_type} von {name}?"
        A = f"{article.capitalize()} {de_type} von {name} ist {data.description}"
        key_string = f"{article} {de_type} von {name}"
        return Q, A, key_string
    else:
        Q = f"What is the {data.description_type} of {data.name}?"
        A = f"The {data.description_type} of {data.name} is {data.description}"
        key_string = f"the {data.description_type} of {data.name}"
        return Q, A, key_string
    
def get_key_string(name: str, description_type: str, language: str) -> str:
    if language == "de":
        # Artikel und Typen-Mapping importieren oder definieren, falls nicht global
        de_type = TYPE_MAPPING.get(description_type, {}).get("de", description_type)
        article = ARTICLE_CASES.get(de_type, {}).get("nominativ", "")
        key_str = f"{article} {de_type} von {clean_name(name)}"
    else:
        key_str = f"the {description_type} of {name}"
    return key_str