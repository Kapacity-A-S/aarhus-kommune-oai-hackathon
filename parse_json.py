import json
from pathlib import Path
from typing import Any, Literal

import instructor
from pydantic import BaseModel, Field
from tqdm import tqdm

import src.llm_extract.azure_client as azure_client


def text_from_table(table: dict[str, Any]) -> str:
    return "\n".join(cell["content"] for cell in table["cells"])


class Diagnosis(BaseModel):
    """A psychiatric diagnosis"""

    name: str = Field(..., title="Danske navn for diagnosen")
    code: str = Field(
        ...,
        title="Kode for diagnosen (Starter med DF, DZ, DX eller DY og følges af et tal)",
    )


event_types = Literal[
    "opvaekst",
    "familieforhold",
    "hjemmets ressourcer",
    "dagligdag i hjemmet",
    "underretning",
]


class extract_case_information(BaseModel):
    """Extracts and summarises all relevant information from a case file. Be as thorough as possible. The more information the better."""

    name: str | None = Field(None, title="Barnets navn")
    diagnoser: list[Diagnosis] | None = Field(None, title="Barnets diagnoser")
    opvaekst: str | None = Field(None, title="Beskrivelse af barnets opvækst")
    familieforhold: str | None = Field(
        None,
        title="Beskrivelse af barnets familieforhold",
    )
    hjemmets_ressourcer: str | None = Field(
        None,
        title="Beskrivelse af hjemmets ressourcer",
    )
    dagligdag_i_hjemmet: str | None = Field(
        None,
        title="Beskrivelse af barnets dagligdag i hjemmet",
    )


class DocumentInfo(BaseModel):
    document_name: str
    start_page: int


class extract_docs(BaseModel):
    documents: list[DocumentInfo]


client = instructor.patch(azure_client.initialize_client())

path = Path("example_data/Case 5.json")
data = json.loads(path.read_text(encoding="utf-8"))

result = data["analyzeResult"]


def extract_content(result: dict, page_number: int = 4) -> str:
    content = "\n".join(
        line["content"] for line in result["pages"][page_number]["lines"]
    )
    return content


first_page = extract_content(result, 0)

doc_info = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Please extract the document information."},
        {"role": "user", "content": first_page},
    ],
    model="gpt-35-turbo",
    temperature=0.0,
    response_model=extract_docs,
)


doc_start_pages = [
    doc.start_page for doc in doc_info.documents if doc.start_page > 0
] + [len(result["pages"]) + 1]

doc_ranges = [
    list(range(start - 1, end - 1))
    for start, end in zip(doc_start_pages, doc_start_pages[1:])
]


contents = [
    "\n".join(extract_content(result, page_number) for page_number in doc_range)
    for doc_range in doc_ranges
]


all_content = "\n".join(
    extract_content(result, page_number) for page_number in range(len(result["pages"]))
)


SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a world class information extraction algorithm. You extract all relevant information from a case file and present it clearly and concisely.",
}


def extract_info_from_doc(all_content: str) -> extract_case_information:
    extracted = client.chat.completions.create(
        messages=[
            SYSTEM_MESSAGE,
            {"role": "user", "content": all_content},
        ],
        model="gpt4-turbo",
        temperature=0.0,
        max_retries=2,
        response_model=extract_case_information,
    )
    return extracted


extractions = [extract_info_from_doc(content) for content in tqdm(contents[1:])]

full_extraction = extract_info_from_doc(all_content)
