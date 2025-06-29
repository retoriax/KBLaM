import argparse
import json
import os
import re
import sys
from itertools import product
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from dotenv import load_dotenv
from tqdm import tqdm
import sys
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time

from langdetect import detect

load_dotenv()

# ---- Define Data Schemas ----
class Entity(BaseModel):
    name: str
    description: str
    objectives: str
    purpose: str

class DataPoint(BaseModel):
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None

# ---- Utility Functions ----
def save_entity(item: BaseModel, filepath: str) -> None:
    """Append a BaseModel as JSON to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a+") as f:
        json.dump(item.model_dump(), f, ensure_ascii=False)
        f.write("\n")


TYPE_MAPPING = {
    "description": {"de": "Beschreibung", "article": "die"},
    "objectives": {"de": "Ziel", "article": "das"},
    "purpose": {"de": "Grund", "article": "der"},
}

def construct_prompts(entity: DataPoint) -> tuple[str, str, str]:
    mapping = TYPE_MAPPING.get(entity.description_type)
    if not mapping:
        raise ValueError(f"Unsupported description_type: {entity.description_type}")

    artikel = mapping["article"]
    typ_de = mapping["de"]
    name = entity.name

    Q = f"Was ist {artikel} {typ_de} von {name}?"
    A = f"{artikel.capitalize()} {typ_de} von {name} ist {entity.description}."
    key_string = f"{artikel} {typ_de} von {name}"

    return Q, A, key_string

# ---- Synthetic Data Generator ----
class SyntheticDataGenerator:
    
    def __init__(self):
        """Initialize the SyntheticDataGenerator with LLM client and prompts."""
        # Initialize ChatOpenAI client
        self.llm = ChatOpenAI(
            model_name=os.getenv("PROXY_MODEL"),
            openai_api_base=os.getenv("PROXY_PATH"),
            openai_api_key=os.getenv("PROXY_API_KEY"),
            temperature=1
        )

        # Prompts schema parser
        self.parser = PydanticOutputParser(pydantic_object=Entity)

        # Source lists (mirroring original script)
        self.idea_sources = [
            "software companies","tech companies","software tools","greek letters",
            "product reviews","product releases","work-related concepts","work-related documents",
            "document types","financial terms","legal terms","medical terms","fiction characters",
            "famous rock bands","birds","animals","natural phenomena","physical locations",
            "artist names","classical music","musical instruments","music genres","art styles",
            "ancient Roman concepts","Hindu myths","Cthulhu Mythos","real-world company names",
            "mythological creatures","planets and stars","historical figures","political figures",
            "literary genres","botanical names","famous landmarks","scientific concepts","space missions",
            "inventions","philosophical terms","chemical elements","famous scientists",
            "famous mathematicians","famous authors","marine life","mythological places",
            "famous battles","sports teams","sport events","food and drinks",
            # Additional examples
            "open source communities","startup culture","science fiction movies","educational platforms",
            "mobile app trends","cybersecurity","quantum computing","sustainability initiatives",
            "climate action groups","metaverse concepts","digital nomad lifestyle","UX/UI patterns",
            "space exploration agencies","blockchain technologies","historical inventions",
            "agile methodologies","robotics startups","biomedical research","online learning tools",
            "sustainable fashion","digital art marketplaces","crowdsourced platforms","voice assistant technologies",
            "urban mobility trends","smart agriculture","AI ethics debates","coding bootcamps",
            "deep learning innovations","human-computer interaction","wearable fitness tech"
        ]
        self.data_types = [
            "person name","idea","team","meeting","event","location","document",
            "presentation","conference","workshop","database","organization","tech company",
            "car company","entertainment company","construction company","retail company",
            "finance company","healthcare company","restaurant","hotel","museum","university",
            "educational institution","government agency","hospital","github repo","project",
            "meeting room","building","product","lab","airline","textbook","tv show",
            "music album","website","personal blog","gaming company","movie studio","consulting firm",
            "biotech company","app","software tool","bookstore","coffee shop","bar","e-commerce site",
            "social media platform","fitness brand","fashion brand","beauty brand","food brand",
            "drink brand","sports brand","travel brand","non-profit organization","political party",
            # Additional examples
            "course platform","mobile app","AI assistant","online service","cloud provider",
            "research paper","whitepaper","community forum","news site","online magazine",
            "technical report","smart device","wearable technology","AR/VR application",
            "AI research lab","open data portal","no-code tool","mental health app","language learning service",
            "podcast platform","virtual workspace","freelancer marketplace","job board","data visualization tool",
            "neural network framework","simulation software","robotic system","telemedicine platform",
            "hardware startup"
        ]

    def perturb_names(self, dataset: List[DataPoint]) -> List[DataPoint]:
        """Perturb names in questions using batching and threading."""
        import time
        def process(dp: DataPoint) -> DataPoint | None:
            def build_prompt(instruction: str):
                return ChatPromptTemplate.from_messages([
                    ("system", "Du bist ein Frage-Perturbator."),
                    ("user", instruction)
                ]) | self.llm

            original_instr = f"Perturbiere den Namen in folgender Frage: '{dp.Q}'. Gib ausschließlich den neuen Fragetext zurück."
            fallback_instr = f"Perturbiere den Namen '{dp.name}' leicht. Gib nur die neue Version zurück."

            max_retries = 7
            for attempt in range(max_retries):
                try:
                    prompt_chain = build_prompt(original_instr)
                    result = prompt_chain.invoke({})
                    if isinstance(result, str):
                        dp.Q = result
                    else:
                        dp.Q = getattr(result, "content", dp.Q)
                    return dp
                except Exception as e:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    print(f"[WARN] Retry {attempt+1} failed for Q: '{dp.Q}'. Waiting {wait:.2f}s. Reason: {e}")
                    time.sleep(wait)
            else:
                # fallback attempt with just the name
                print(f"[ERROR] Max retries reached for original Q: '{dp.Q}'. Attempting fallback via name...")
                try:
                    prompt_chain = build_prompt(fallback_instr)
                    result = prompt_chain.invoke({})
                    if isinstance(result, str):
                        dp.Q = result
                    else:
                        dp.Q = getattr(result, "content", dp.Q)
                    return dp
                except Exception as e:
                    print(f"[FATAL] Fallback failed for name '{dp.name}': {e}")
                    with open("perturbation_failures.log", "a") as logf:
                        logf.write(f"{dp.Q}\n")
                    return None

        batch_size = 16
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

        perturbed = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(lambda b: [res for dp in b if (res := process(dp))], batch): batch for batch in batches}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Perturbiere Namen",
                dynamic_ncols=False,
                ascii=True,
                mininterval=0.5,
                ncols=80,
                smoothing=0,
                disable=False,
                file=sys.stderr
            ):
                try:
                    result = future.result()
                    perturbed.extend(result)
                except Exception as e:
                    print(f"[ERROR] Fehler in Batch: {e}")
        return perturbed

    def set_seed(self, seed: int) -> None:
        """Stub for seed compatibility; ChatOpenAI does not support seed directly."""
        self.seed = seed

    def get_instructions(self) -> List[str]:
        """Generate raw prompts for entity name creation."""
        return [
            f"Please randomly generate a {dtype} name innovated by or associated with {source}."
            for source, dtype in product(self.idea_sources, self.data_types)
        ]
    
    def get_instructions_de(self) -> List[str]:
        """Generate raw prompts for entity name creation."""
        return [
            f"Bitte generiere zufällig einen Namen für ein(e) {dtype}, der mit {source} assoziiert ist oder daraus hervorgegangen sein könnte."
            for source, dtype in product(self.idea_sources, self.data_types)
]

    def generate_entity(self, instruction: str) -> Entity:
        """Generate an Entity with name, description, objectives, purpose via structured LLM output."""
        # Build prompt with format instructions
        raw_fmt = self.parser.get_format_instructions()
        fmt = raw_fmt.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI system that generates synthetic data examples in JSON format. You provide only German synthetic data, even if the prompts are in English. Use Beschreibung, Ziel, Grund instead of description, objective, purpose. Use correct German grammar."),
            ("user", instruction + "\nFormat instructions:\n" + fmt)
        ])
        # Compose chain
        chain = prompt | self.llm.with_structured_output(Entity)
        return chain.invoke({})

    def generate_entity_de(self, instruction: str) -> Entity:
        """Generate an Entity with name, description, objectives, purpose via structured LLM output in German."""
        raw_fmt = self.parser.get_format_instructions()
        fmt = raw_fmt.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein KI-System, das synthetische Beispieldaten auf Deutsch im JSON-Format erzeugt. Verwende ausschließlich deutsche Begriffe und achte auf korrekte Grammatik. Nutze 'Beschreibung', 'Ziel', 'Grund'."),
            ("user", instruction + "\nFormatvorgaben:\n" + fmt)
        ])
        chain = prompt | self.llm.with_structured_output(Entity)
        return chain.invoke({})
    
    def generate_related_data(self, ent: Entity) -> Entity:
        """Generate a related Entity (e.g. person) using structured output."""
        instruction = (
            f"Generate a person name related to the entity {ent.name} with description '{ent.description}'."
            " Make sure to use the same JSON format."
        )
        raw_fmt = self.parser.get_format_instructions()
        fmt = raw_fmt.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI system that generates related entities in JSON format."),
            ("user", instruction + "\nFormat instructions:\n" + fmt)
        ])
        chain = prompt | self.llm.with_structured_output(Entity)
        return chain.invoke({})
    
    def generate_related_data_de(self, ent: Entity) -> Entity:
        """Generate a related Entity (e.g. person) using structured output in German."""
        instruction = (
            f"Erzeuge eine Person, die mit der Entität {ent.name} mit der Beschreibung '{ent.description}' in Verbindung steht. "
            "Nutze dasselbe JSON-Format mit den Feldern 'name', 'Beschreibung', 'Ziel', 'Grund'."
        )
        raw_fmt = self.parser.get_format_instructions()
        fmt = raw_fmt.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein KI-System, das verwandte Entitäten im JSON-Format auf Deutsch erzeugt."),
            ("user", instruction + "\nFormatvorgaben:\n" + fmt)
        ])
        chain = prompt | self.llm.with_structured_output(Entity)
        return chain.invoke({})

    def post_process_data(self, entities: List[Entity]) -> List[DataPoint]:
        """Convert raw Entity objects into QA DataPoints."""
        dataset: List[DataPoint] = []
        for ent in entities:
            for field in ["description", "objectives", "purpose"]:
                dp = DataPoint(
                    name=ent.name,
                    description_type=field,
                    description=getattr(ent, field)
                )
                dp.Q, dp.A, dp.key_string = construct_prompts(dp)
                dataset.append(dp)
        return dataset

    def augment_qa(self, dataset: List[DataPoint], filepath: str) -> List[DataPoint]:
        """Parallel augment QA pairs using ThreadPoolExecutor in batches of 8.
        Writes each successful augmentation directly to file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        output_file = open(filepath, "a", encoding="utf-8")
        def process(dp: DataPoint) -> DataPoint | None:
            instr = (
                f"Gegeben ist das Frage-Antwort-Paar:\n"
                f"Frage: {dp.Q}\n"
                f"Antwort: {dp.A}\n"
                "Formuliere eine offene, weiterführende Frage zu diesem Paar und gib die neue Frage und eine vollständige, passende Antwort im Format 'Q: ...' 'A: ...' zurück. "
                "Beide Teile müssen enthalten sein. Verwende ausschließlich korrektes Deutsch."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a QA augmenter."),
                ("user", instr)
            ])
            chain = prompt | self.llm
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    content = chain.invoke({})
                    break
                except Exception:
                    backoff = 2 ** attempt
                    print(f"Rate limit hit, retrying in {backoff} seconds...")
                    time.sleep(backoff)
            else:
                print("[ERROR] Max retries reached.")
                return None
            text = content.content
            match_q = re.search(r"Q: (.*)", text)
            match_a = re.search(r"A: (.*)", text)
            if match_q and match_a:
                dp.extended_Q = match_q.group(1)
                dp.extended_A = match_a.group(1) if match_a.group(1).strip() else "[Antwort fehlt]"
                # Write to file immediately
                json.dump(dp.model_dump(), output_file, ensure_ascii=False)
                output_file.write("\n")
                return dp
            else:
                print(f"[WARN] Unexpected format in response: {text}")
                return None

        def process_batch(batch: List[DataPoint]) -> List[DataPoint]:
            results = []
            for dp in batch:
                res = process(dp)
                if res:
                    results.append(res)
            return results

        batch_size = 4
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

        augmented = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Augmentiere QA",
                dynamic_ncols=False,
                ascii=True,
                mininterval=0.5,
                ncols=80,
                smoothing=0,
                disable=False,
                file=sys.stderr
            ):
                result = future.result()
                if result:
                    augmented.extend(result)
        output_file.close()
        return augmented

def _generate_and_save(instruction: str, gen: SyntheticDataGenerator, raw_fp: str, generate_related: bool) -> list[Entity]:
    """Generate entity (and related if enabled), save to raw_fp, and return list of entities.
    Skips saving if description is not detected as German."""
    results = []
    ent = gen.generate_entity_de(instruction)
    try:
        lang = detect(ent.description)
    except Exception as e:
        print(f"[WARN] Language detection failed for entity '{ent.name}': {e}")
        lang = "unknown"
    if lang != "de":
        print(f"[SKIP] English entity detected:\n{ent.model_dump_json(indent=2, ensure_ascii=False)}")
    else:
        save_entity(ent, raw_fp)
        results.append(ent)
        if generate_related:
            rel = gen.generate_related_data_de(ent)
            try:
                rel_lang = detect(rel.description)
            except Exception as e:
                print(f"[WARN] Language detection failed for related entity '{rel.name}': {e}")
                rel_lang = "unknown"
            if rel_lang != "de":
                print(f"[SKIP] English related entity detected:\n{rel.model_dump_json(indent=2, ensure_ascii=False)}")
            else:
                save_entity(rel, raw_fp)
                results.append(rel)
    return results

def _generate_batch_and_save(batch_instructions: List[str], gen: SyntheticDataGenerator, raw_fp: str, generate_related: bool) -> List[Entity]:
    """Generate and save a batch of entities."""
    results = []
    for instr in batch_instructions:
        ents = _generate_and_save(instr, gen, raw_fp, generate_related)
        results.extend(ents)
    return results

# ---- Argument Parsing ----
def parser_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation with ChatOpenAI")
    parser.add_argument("--output_path", type=str, default="dataset")
    parser.add_argument("--raw_output_file", type=str, default="synthetic_data_raw_ger.json")
    parser.add_argument("--output_file", type=str, default="synthetic_data_QA_ger.json")
    parser.add_argument("--augmented_output_file", type=str, default="synthetic_data_QA_augmented_ger.json")
    parser.add_argument("--perturbed_output_file", type=str, default="synthetic_data_QA_perturbed_ger.json")
    parser.add_argument("--generate_related_people", action="store_true", default=True)
    return parser.parse_args()

# ---- Main Execution ----
if __name__ == "__main__":
    args = parser_args()
    os.makedirs(args.output_path, exist_ok=True)

    gen = SyntheticDataGenerator()
    raw_fp = os.path.join(args.output_path, args.raw_output_file)
    qa_fp  = os.path.join(args.output_path, args.output_file)
    aug_fp = os.path.join(args.output_path, args.augmented_output_file)
    per_fp = os.path.join(args.output_path, args.perturbed_output_file)

    entities: List[Entity] = []
    # Generate or load raw entities
    if os.path.exists(raw_fp):
        # Lade bestehende rohe Daten mit Fortschrittsanzeige
        # Zähle zunächst die Zeilen für die Gesamtlänge
        with open(raw_fp, "r") as f:
            total_lines = sum(1 for _ in f)
        # Lese und verarbeite jede Zeile mit Fortschrittsbalken
        with open(raw_fp, "r") as f:
            for line in tqdm(
                f,
                total=total_lines,
                desc="Lade rohe Entities",
                dynamic_ncols=False,
                ascii=True,
                mininterval=0.5,
                ncols=80,
                smoothing=0,
                disable=False,
                file=sys.stderr
            ):
                entities.append(Entity.model_validate_json(line))
    else:
        for seed in range(3):
            gen.set_seed(seed)
            instructions = gen.get_instructions_de()
            batch_size = 8
            batches = [instructions[i:i+batch_size] for i in range(0, len(instructions), batch_size)]
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(_generate_batch_and_save, batch, gen, raw_fp, args.generate_related_people): batch
                    for batch in batches
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Erzeuge Batches",
                    dynamic_ncols=False,
                    ascii=True,
                    mininterval=0.5,
                    ncols=80,
                    smoothing=0,
                    disable=False,
                    file=sys.stderr
                ):
                    try:
                        batch_ents = future.result()
                        entities.extend(batch_ents)
                    except Exception as e:
                        batch = futures[future]
                        print(f"Error generating batch {batch!r}: {e}")

    # Post-process into QA
    datapoints = gen.post_process_data(entities)
    for dp in datapoints:
        save_entity(dp, qa_fp)

    # Perturb names and save perturbed questions
    # perturbed = gen.perturb_names(datapoints)
    # for dp in perturbed:
    #     save_entity(dp, per_fp)

    # Augment QA (wenn Datei nicht vorhanden)
    if os.path.exists(aug_fp):
        augmented = []
        with open(aug_fp, "r") as f:
            total_lines = sum(1 for _ in f)
        skipped = 0
        with open(aug_fp, "r") as f:
            for i, line in enumerate(tqdm(
                f,
                total=total_lines,
                desc="Lade augmentierte QA",
                dynamic_ncols=False,
                ascii=True,
                mininterval=0.5,
                ncols=80,
                smoothing=0,
                disable=False,
                file=sys.stderr
            )):
                try:
                    augmented.append(DataPoint.model_validate_json(line))
                except Exception as e:
                    skipped += 1
                    print(f"[SKIP] Fehlerhafte JSON-Zeile {i}: {e}")
                    with open("broken_augmented_lines.log", "a", encoding="utf-8") as logf:
                        logf.write(f"{line.strip()}\n")
        if skipped > 0:
            print(f"⛔️ {skipped} Zeilen konnten nicht geladen werden (siehe broken_augmented_lines.log)")
    else:
        augmented = gen.augment_qa(datapoints, aug_fp)

    # Save all augmented datapoints as a JSON array
    full_augmented_array_fp = os.path.join(args.output_path, "synthetic_data_ger.json")
    with open(full_augmented_array_fp, "w") as f:
        json.dump([dp.model_dump() for dp in augmented], f, ensure_ascii=False, indent=2)