import gradio as gr
import llm_extract.azure_client as azure_client
import llm_extract.pydantic_extraction as pydantic_extraction
from pydantic import BaseModel, Field

OAI_CLIENT = azure_client.initialize_client()

class extract_fairy_tale_information(BaseModel):
    """Extract high-level information about a fairy tale"""
    story_title: str = Field(..., title="Title of the fairy tale")
    author_name: str | None = Field(None, title="Author of the fairy tale")
    important_characters: list[str] = Field(..., title="Important characters in the fairy tale")
    one_sentence_summary: str = Field(..., title="One sentence summary of the fairy tale")


def classify_fairy_tale(FairyTale: str) -> tuple[str, str, str, str]:
    """Classify a fairy tale using the OpenAI API"""
    output_schema = pydantic_extraction.extract_with_schema(client=OAI_CLIENT, document=FairyTale, schema=extract_fairy_tale_information)
    return output_schema.story_title, output_schema.author_name, ", ".join(output_schema.important_characters), output_schema.one_sentence_summary

gr.Text(label="Title")

demo = gr.Interface(
    fn=classify_fairy_tale, inputs="text", outputs=[gr.Text(label="Title"), gr.Text(label="Author"),gr.Text(label="Important Characters"),gr.Text(label="Summary")], title="Fairy Tale Classifier")

demo.launch()
