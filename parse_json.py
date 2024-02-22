import datetime
import json
from pathlib import Path
from pprint import pprint
from typing import Literal

import instructor
from pydantic import BaseModel, Field

import src.llm_extract.azure_client as azure_client


def text_from_table(table):
    return "\n".join(cell["content"] for cell in table["cells"])


class Diagnosis(BaseModel):
    """A psychiatric diagnosis"""
    name: str = Field(..., title="Danske navn for diagnosen")
    code: str = Field(..., title="Kode for diagnosen (Starter med DF, DZ, DX eller DY og følges af et tal)")



event_types = Literal["opvaekst", "familieforhold", "hjemmets ressourcer", "dagligdag i hjemmet", "underretning"]

    




class extract_case_information(BaseModel):
    """Extracts and summarises all relevant information from a case file. Be as thorough as possible. The more information the better."""
    name: str = Field(..., title="Barnets navn")
    opvaekst: str = Field(..., title="Beskrivelse af barnets opvækst")
    familieforhold: str = Field(..., title="Beskrivelse af barnets familieforhold")
    hjemmets_ressourcer: str = Field(..., title="Beskrivelse af hjemmets ressourcer")
    dagligdag_i_hjemmet: str = Field(..., title="Beskrivelse af barnets dagligdag i hjemmet")
    underretninger: list[str] = Field(..., title="Underretninger om barnet")




client = instructor.patch(azure_client.initialize_client())

path = Path("example_data/Case 5.json")
data = json.loads(path.read_text(encoding="utf-8"))

result = data["analyzeResult"]

def extract_content(result: dict, page_number: int = 4) -> str:
    content = "\n".join(line["content"] for line in result["pages"][page_number]["lines"])
    return content


len(result["pages"])

bfu_range = range(7, 24)
bfu_content = "\n".join(extract_content(result, page_number) for page_number in bfu_range)

content = extract_content(result)

SYSTEM_MESSAGE = {"role": "system", "content": "You are a world class information extraction algorithm. Your accuracy, precision and recall are all 100%. You only extract information from the given text, never add any of your own."}

extracted = client.chat.completions.create(
    messages=[
        SYSTEM_MESSAGE,
        {"role": "user", "content": bfu_content}
    ],
    model="gpt-35-turbo",
    response_model=extract_case_information
)


pprint(extracted.model_dump())