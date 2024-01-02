"""
This has been greatly inspired by the below blog post:
https://minimaxir.com/2023/12/chatgpt-structured-data/
"""
import json

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

SYSTEM_PROMPT = "You are a state-of-the-art data extraction algorithm. Every time you extract a correct piece of information you get $100 in tips."


def _parse_oai_output(output: ChatCompletion, model: type[BaseModel]) -> BaseModel:
    return model(**json.loads(output.choices[0].message.function_call.arguments))


def schema_to_function(schema: type[BaseModel]) -> dict:
    assert schema.__doc__, f"{schema.__name__} is missing a docstring."
    assert (
        "title" not in schema.model_fields
    ), "`title` is a reserved keyword and cannot be used as a field name."
    schema_dict = schema.model_json_schema()
    schema_dict.pop("description")
    schema_dict.pop("title")
    return {
        "name": schema.__name__,
        "description": schema.__doc__,
        "parameters": schema_dict,
    }


def extract_with_schema(
    client: AzureOpenAI,
    document: str,
    schema: type[BaseModel],
    system_prompt: str = SYSTEM_PROMPT,
    model: str = "gpt4",
) -> BaseModel:
    """Extracts information from a document using a schema.

    This function uses the provided schema to extract information from a document.
    It formats the document, generates output using the AzureOpenAI client and the
    provided model, and then parses the output using the provided schema.

    Args:
        client (AzureOpenAI): The AzureOpenAI client to use for generating output.
        document (str): The document from which to extract information.
        schema (type[BaseModel]): The Pydantic schema to use for information extraction.
        system_prompt (str, optional): The system prompt to use. Defaults to SYSTEM_PROMPT.
        model (str, optional): The mode-name to use for generating output. Defaults to "gpt4". Must be deployed on Azure!

    Returns:
        BaseModel: The extracted information, parsed using the provided schema.

    """
    function_schema = schema_to_function(schema=schema)
    messages = _format_document(document=document, system_prompt=system_prompt)
    output = _generate_output(
        client,
        messages,
        model_name=model,
        function_schema=function_schema,
    )
    return _parse_oai_output(output=output, model=schema)


def _generate_output(
    client: AzureOpenAI,
    messages: list[ChatCompletionMessageParam],
    function_schema: dict,
    model_name: str,
) -> ChatCompletion:
    output = client.chat.completions.create(
        messages=messages,
        temperature=0.0,
        model=model_name,
        functions=[function_schema],
        function_call={
            "name": function_schema["name"],
        },
    )
    return output


def _format_document(
    document: str,
    system_prompt: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"```document\n{document}\n```"},
    ]
