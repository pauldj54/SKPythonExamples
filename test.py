import semantic_kernel as sk
import os
from pydantic import ValidationError
kernel = sk.Kernel()

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.curdir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

try:
    # this will use the given values as the settings
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    print(api_key)
    # service = OpenAIChatCompletion(
    #     service_id="openai_chat_service",
    #     api_key=api_key,
    #     ai_model_id="gpt-4o",
    # )
except ValidationError   as e:
    print(e)