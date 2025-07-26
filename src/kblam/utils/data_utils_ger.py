from kblam.utils.ger_utils import clean_name, TYPE_MAPPING, ARTICLE_CASES
from kblam.utils.data_utils import DataPoint, Entity, load_entities, save_entity

import numpy as np

def get_i_dont_know_ans():
    # return "I am sorry I cannot find relevant information in the KB."
    return "Es tut mir leid, ich kann in der Wissensdatenbank keine passenden Informationen finden."


def augment_row(row: dict[str, str]) -> list[dict[str, str]]:
    """Augment an entity with questions from pre-defined templates."""
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
