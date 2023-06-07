import os
import json
import datetime
from typing import Any

from nemoguardrails.rails import LLMRails, RailsConfig

from langchain.chat_models import ChatOpenAI
import openai

import replicate
from firebase_admin import storage

from .common.utils import (
    OPENAI_API_KEY,
    FIREBASE_STORAGE_ROOT,
)
from .image_embedding import (
    query_image_text,
    get_prompt_image_with_message,
)

# Give the path to the folder containing the rails
file_path = os.path.dirname(os.path.abspath(__file__))
config = RailsConfig.from_path(f"{file_path}/guardrails-config")


def getCompletion(
    query,
    model="gpt-3.5-turbo",
    uuid="",
    image_search=True,
):
    llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=OPENAI_API_KEY)
    app = LLMRails(config, llm)

    message = app.generate(messages=[{"role": "user", "content": query}])
    return message["content"]


def query_image_ask(image_content, message, uuid):
    prompt_template = get_prompt_image_with_message(image_content, message)
    data = getCompletion(query=prompt_template, uuid=uuid, image_search=False)
    # chain_data = json.loads(data.replace("'", '"'))
    chain_data = json.loads(data)
    if chain_data["program"] == "image":
        return True
    return False


def getTextFromImage(filename):
    # Create a reference to the image file you want to download
    bucket = storage.bucket()
    blob = bucket.blob(FIREBASE_STORAGE_ROOT.__add__(filename))
    download_url = ""

    try:
        # Download the image to a local file
        download_url = blob.generate_signed_url(
            datetime.timedelta(seconds=300), method="GET", version="v4"
        )

        output = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={"image": download_url},
        )

    except Exception as e:
        output = str("Error happend while analyzing your prompt. Please ask me again :")

    return str(output)


"""chat with ai
response: 
{
 'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-3.5-turbo',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'},
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
"""


# Define a content filter function
def filter_guardrails(model: any, query: str):
    llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=OPENAI_API_KEY)
    app = LLMRails(config, llm)

    # get message from guardrails
    message = app.generate(messages=[{"role": "user", "content": query}])

    if (
        json.loads(message["content"])["content"]
        == "Sorry, I cannot comment on anything which is relevant to the password or pin code."
        or message["content"]
        == "I am an Rising AI assistant which helps answer questions based on a given knowledge base."
    ):
        return json.loads(message["content"])["content"]
    else:
        return ""


def handle_chat_completion(messages: Any, model: str = "gpt-3.5-turbo") -> Any:
    openai.api_key = OPENAI_API_KEY

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Filter the reply using the content filter
    result = filter_guardrails(model, messages[-1]["content"])

    if result == "":
        return response
    else:
        response["choices"][0]["message"]["content"] = result
        return response
