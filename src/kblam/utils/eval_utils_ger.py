from kblam.utils.eval_utils import (
    _prune_for_llama,
    _prune_for_phi3,
    _format_Q_llama,
    _format_Q_phi3,
    softmax,
    model_question_format_mapping,
    model_prune_format_mapping,
    answer_question,
)
from kblam.utils.ger_utils import (
    get_article
)

instruction_prompt: str
instruction_prompt_multi_entities: str
zero_shot_prompt: str
zero_shot_prompt_multi_entities: str

instruction_prompts_ger = """
Bitte beantworten Sie Fragen auf Grundlage des gegebenen Textes im Format: "{article} {property} von {name} ist {description}"
"""

instruction_prompts_multi_entities = """
Bitte beantworten Sie Fragen auf Grundlage des gegebenen Textes im Format: "{article}_1 {property}_1 von {name}_1 ist {description}; {article}_2 {property}_2 von {name}_2 ist {description}; ..."
"""

zero_shot_prompt = """
Bitte beantworten Sie die Frage in sehr kompakter Form im Format: {article} {property} von {name} ist {description}
"""

zero_shot_prompt_multi_entities = """
Bitte beantworten Sie die Frage in sehr kompakter Form im Format: "{article} {property} von {name1} ist {description}; {article} {property} von {name2} ist {description}; ..."
"""
