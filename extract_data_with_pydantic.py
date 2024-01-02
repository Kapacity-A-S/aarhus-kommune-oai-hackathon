import pprint

import common.openai_helper_functions.azure_client as azure_client
import common.openai_helper_functions.pydantic_extraction as pydantic_extraction
from loguru import logger
from pydantic import BaseModel, Field


class extract_song_info(BaseModel):
    """Extract information about a song from a page"""

    song_title: str = Field(..., description="The name of the song")
    artist: str = Field(
        ...,
        description="The name of the artist or band who performed the song",
    )
    album: str = Field(None, description="The name of the album that contains the song")
    genre: list[str] = Field(None, description="The musical genre of the song")
    duration: float = Field(..., description="The length of the song in seconds", gt=0)
    release_year: int = Field(
        ...,
        description="The year when the song was released",
        ge=1900,
        le=2023,
    )


if __name__ == "__main__":
    TEXT = """
"Teenage Dirtbag"

Single by Wheatus
from the album Wheatus
Released	June 20, 2000
Recorded	February–March 2000[1]
Genre
Pop rock[2]alternative pop[3]pop punk[4]
Length	4:07
Label	Columbia
Songwriter(s)	Brendan B. Brown
Producer(s)
WheatusPhilip A. Jimenez
Wheatus singles chronology
"Teenage Dirtbag"
(2000)	"A Little Respect"
(2001)

Music video
"Teenage Dirtbag" on YouTube

    """.strip()  # noqa: RUF001

    logger.info(f"Extracting song information from: {pprint.pformat(TEXT)}")

    # initalize the client (remember to set the environment variables from the .env_example file!)
    client = azure_client.initialize_client()

    logger.info(
        f"Using schema: {pprint.pformat(extract_song_info.model_json_schema())}",
    )

    # extract the information from the text
    logger.info("Parsing text!")
    parsed_output = pydantic_extraction.extract_with_schema(
        client=client,
        document=TEXT,
        schema=extract_song_info,
    )
    logger.info(pprint.pformat(parsed_output.model_dump()))
