def clean_name(name: str) -> str:
    parts = name.split()
    if parts[0].lower() in {"der", "die", "das"}:
        return " ".join(parts[1:])
    return name

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

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Entity:
    name: str
    description: str
    objectives: str
    purpose: str


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


def save_entity(pair: Entity | DataPoint, output_file: str) -> None:
    """Save a JSON entity to a file."""
    try:
        with open(output_file, "a+") as f:
            json.dump(pair.__dict__, f)
            f.write("\n")
    except Exception as e:
        print("Error saving entity.")
        print(e)


def load_entities(inout_file: str) -> list[Entity | DataPoint]:
    """Load entities from a file."""
    entities = []
    try:
        with open(inout_file, "r") as f:
            for line in f:
                entity = json.loads(line)
                entities.append(entity)
    except Exception as e:
        print("Error loading entities.")
        print(e)
    return entities


def get_i_dont_know_ans():
    # return "I am sorry I cannot find relevant information in the KB."
    return "Es tut mir leid, ich kann in der Wissensdatenbank keine passenden Informationen finden."


def augment_row(row: dict[str, str]) -> list[dict[str, str]]:
    """Augment an entity with questions from pre-defined templates."""
    # templates = [
    #     "What {} does {} have?",
    #     "What is the {} of {}?",
    #     "Tell me about the {} of {}.",
    #     "Can you let me know the {} of {}?",
    #     "Can you inform me about the {} of {}?",
    #     "Describe the {} of {}.",
    #     "What details can you share about the {} of {}?",
    #     "What kind of {} does {} have?",
    #     "Provide details on the {} of {}.",
    #     "What features does the {} of {} include?",
    #     "Can you elaborate on the {} of {}?",
    #     "How would you describe the {} of {}?",
    #     "What can you tell me about the {} characteristics of {}?",
    #     "Can you explain the {} of {}?",
    #     "What insights can you provide about the {} of {}?",
    #     "What should I know about the {} of {}?",
    # ]
    templates = [
        ("Was ist {article} {type} von {name}?", "nominativ"),
        ("Erzähle mir von {article} {type} von {name}.", "dativ"),
        ("Kannst du mir {article} {type} von {name} nennen?", "akkusativ"),
        ("Was kannst du mir über {article} {type} von {name} sagen?", "akkusativ"),
        ("Beschreibe {article} {type} von {name}.", "akkusativ"),
        ("Was sollte ich über {article} {type} von {name} wissen?", "akkusativ"),
        ("Welche Eigenschaften hat {article} {type} von {name}?", "nominativ"),
        ("Welche Informationen gibt es zu {article} {type} von {name}?", "dativ"),
        ("Kannst du mir Einzelheiten zu {article} {type} von {name} geben?", "dativ"),
        ("Wie würdest du {article} {type} von {name} beschreiben?", "akkusativ"),
        ("Was kannst du mir über die Merkmale {genitiv_article} {type} von {name} sagen?", "genitiv"),
        ("Kannst du mir {article} {type} von {name} genauer erklären?", "akkusativ"),
        ("Welche Erkenntnisse gibt es über {article} {type} von {name}?", "akkusativ"),
    ]
    dtype = row["description_type"]
    name = clean_name(row["name"])
    tid = np.random.randint(0, len(templates))
    template, case = templates[tid]
    de_type = TYPE_MAPPING.get(dtype, {}).get("de", dtype)
    article = ARTICLE_CASES.get(de_type, {}).get(case, "")
    article_genitiv = ARTICLE_CASES.get(de_type, {}).get("genitiv", "")
    return template.format(article=article, genitiv_article=article_genitiv, type=de_type, name=name)


def generate_multi_entity_qa(
    names: list[str], properties: list[str], answers: list[str]
) -> tuple[str, str]:
    """Generate a question-answer pair for multiple entities."""
    # templates = [
    #     "What is {}?",
    #     "Tell me {}.",
    #     "Can you let me know {}?",
    #     "Can you inform me {}?",
    #     "Describe {}.",
    #     "Explain {}.",
    #     "Could you describe the {}?",
    #     "What can you tell me about {}?",
    #     "Could you provide information on {}?",
    #     "Please enlighten me about {}.",
    #     "Can you clarify {} for me?",
    #     "Could you give me a detailed description of {}?",
    #     "I need more information on {}.",
    # ]
    templates = [
        ("Was ist {0}?", "nominativ"),
        ("Erzähle mir von {0}.", "dativ"),
        ("Kannst du mir bitte von {0} erzählen?", "dativ"),
        ("Beschreibe bitte {0}.", "akkusativ"),
        ("Kannst du mir {0} näher erklären?", "akkusativ"),
        ("Ich brauche mehr Informationen über {0}.", "akkusativ"),
    ]
    template_idx = np.random.randint(0, len(templates))
    template, case = templates[template_idx]
    question_body = ""
    # Use TYPE_MAPPING for German article and type
    for raw_name, property in zip(names[:-1], properties[:-1]):
        name = clean_name(raw_name)
        de_type = TYPE_MAPPING.get(property, {}).get("de", property)
        article = ARTICLE_CASES.get(de_type, {}).get(case, "")
        question_body += f"{article} {de_type} von {name}, "
    # Last iteration with "und"
    de_type = TYPE_MAPPING.get(properties[-1], {}).get("de", properties[-1])
    article = ARTICLE_CASES.get(de_type, {}).get(case, "")
    name = clean_name(names[-1])
    question_body = question_body.rstrip(", ") + f" und {article} {de_type} von {name}"
    answer_str = ""
    for answer, raw_name, property in zip(answers, names, properties):
        name = clean_name(raw_name)
        de_type = TYPE_MAPPING.get(property, {}).get("de", property)
        article = ARTICLE_CASES.get(de_type, {}).get("nominativ", "")
        answer_str += f"{article.capitalize()} {de_type} von {name} ist {answer}; "

    return template.format(question_body), answer_str.strip()
