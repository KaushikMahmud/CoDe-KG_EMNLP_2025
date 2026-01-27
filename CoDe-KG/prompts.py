# prompts.py

COREf_FICL_SYSTEM = r"""You are a coreference resolution agent. Below is a biomedical abstract presented as tokenized text with indices. Your task is to identify and annotate coreference expressions within the text. For each co-referent expression:

- Record the surface form under "Expression".
- Use the provided token indices as "StartToken" and "EndToken" (they are the same for single-token expressions).
- Map each expression to its antecedent using "RefersTo" --- either a noun phrase or named entity from the text.
- Only include pronouns or repeated noun phrases referring back to a prior concept or entity.

Use this format:
{
"Expression": "string",
"StartToken": int,
"EndToken": int,
"RefersTo": "string"
}
"""

COREf_FICL_USER = r"""Now process this tokenized abstract:

{tokenized_text}
"""


SIMPLIFY_COMPLEX_SYSTEM = r"""Below is a step-by-step process. For each example, think step by step, then output only the simplified sentences in the form:
S1 -> ...
S2 -> ...
...
one per line, and nothing else.

Example 1:
Input:
“A prospective cohort study was conducted in Leeds, UK, based on routinely collected data from a service that allowed patients with symptoms of lung cancer to request CXR.”
Output:
S1 -> A prospective cohort study was conducted in Leeds, UK.
S2 -> The study was based on routinely collected data from a service.
S3 -> The service allowed patients with symptoms of lung cancer to request CXR.

Example 2:
Input:
“After the cells were treated with the drug, which had been synthesized in our lab, we measured the change in fluorescence using a spectrophotometer.”
Output:
S1 -> We measured the change in fluorescence using a spectrophotometer.
S2 -> The cells were treated with the drug.
S3 -> The drug had been synthesized in our lab.

Now apply the same process to this new sentence.

***OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else.***
"""

SIMPLIFY_COMPLEX_USER = r"""Input: "{sentence}"
"""


SIMPLIFY_COMPOUND_SYSTEM = r"""Below is a process to convert a compound sentence into simple sentences.
For each example, think step by step, then output only the simplified sentences in the form:
S1 -> ...
S2 -> ...
...
one per line, and nothing else.

Example 1:
Input:
“Lung cancer stands prominently among the foremost contributors to human mortality, distinguished by its elevated fatality rate and the second-highest incidence rate among malignancies, and the metastatic dissemination of lung cancer stands as a primary determinant of its elevated mortality and recurrence rates.”
Output:
S1 -> Lung cancer stands prominently among the foremost contributors to human mortality.
S2 -> It is distinguished by its elevated fatality rate and the second-highest incidence rate among malignancies.
S3 -> The metastatic dissemination of lung cancer stands as a primary determinant of its elevated mortality and recurrence rates.

Example 2:
Input:
“Climate change accelerates the melting of polar ice, and rising sea levels threaten coastal communities around the world.”
Output:
S1 -> Climate change accelerates the melting of polar ice.
S2 -> Rising sea levels threaten coastal communities around the world.

Now apply the same process to this new sentence.

OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else.
"""

SIMPLIFY_COMPOUND_USER = r"""Input: "{sentence}"
"""

SIMPLIFY_COMPOUND_COMPLEX_SYSTEM = r"""Below is a process to split a compound-complex sentence into standalone simple sentences. Think step by step, then apply.

Example 1:
Input:
“Although lung cancer is the leading cause of US cancer-related deaths, lung cancer screening with a low radiation dose chest computed tomography scan is now standard of care for a high-risk eligible population, and clinicians and surgeons must evaluate the trade-offs of benefits and harms, including the identification of many benign lung nodules, overdiagnosis, and complications.”
Output:
S1 -> Lung cancer is the leading cause of US cancer-related deaths.
S2 -> Lung cancer screening with a low-dose chest computed tomography scan is now standard of care for a high-risk eligible population.
S3 -> Lung cancer screening is recommended for a high-risk, eligible population.
S4 -> Clinicians and surgeons must evaluate the trade-offs of benefits and harms.
S5 -> Evaluated trade-offs include the identification of many benign lung nodules.
S6 -> Evaluated trade-offs include the risk of overdiagnosis.
S7 -> Evaluated trade-offs include complications from lung-cancer screening.

Example 2:
Input:
“Although warmed by the sun, the fields remained dry, and farmers worried about the drought.”
Output:
S1 -> The sun warmed the fields.
S2 -> The fields remained dry.
S3 -> Farmers worried about the drought.

Now apply to this new sentence.

OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else.
"""

SIMPLIFY_COMPOUND_COMPLEX_USER = r"""Input: "{sentence}"
"""


REL_EXTRACT_SYSTEM = r"""You are a knowledge graph relationship extraction agent. Your task is to extract structured relationships from simple sentences to create knowledge graph triples. Each triple should contain two entities and the relationship between them.

- Analyze the sentence structure and identify key components.
- Extract all meaningful entities (nouns, noun phrases, proper nouns, concepts).
- Identify relationships between entities based on verbs, prepositions, and semantic meaning.
- Form triples (Entity 1 -> Relationship -> Entity 2) as structured relationships.
- Validate that each triple captures meaningful semantic information.

Examples:

Input:
"Regulating miR-497-5p provides a potential targeted therapy for lung cancer treatment."
Output:
[{
"Entity 1": "regulating miR-497-5p",
"Entity 2": "lung cancer targeted treatment",
"Relationship": "provides"
}]

Input:
"The activation of caspase signal pathway was the reason for stronger apoptosis."
Output:
[{
"Entity 1": "activation of caspase signal pathway",
"Entity 2": "stronger apoptosis",
"Relationship": "was the reason for"
}]

Input:
"With clinical significance features selection, over-sampling methods achieved the highest AUC results."
Output:
[{
"Entity 1": "clinical significance features selection",
"Entity 2": "over-sampling methods",
"Relationship": "With"
},
{
"Entity 1": "over-sampling methods",
"Entity 2": "highest AUC results",
"Relationship": "achieved"
}]
"""

REL_EXTRACT_USER = r"""Now extract knowledge graph relationships from this sentence: {sentence}
"""


def build_prompt(system_text: str, user_text: str, **kwargs) -> str:
    return system_text.strip() + "\n\n" + user_text.format(**kwargs).strip()
