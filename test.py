import asyncio, os, logging

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Get keys and configuration from .env file
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.curdir, '.env')
load_dotenv(dotenv_path)

# Azure OpenAI Details
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_deployment_name = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_NAME")

# Main function (needed to make async methods work)
async def main()-> None:
    # Configure the kernel
    kernel = Kernel()

    gpt4 = AzureChatCompletion(
        api_key = api_key, 
        endpoint = api_endpoint, 
        deployment_name = "gpt-4",
        service_id= 'gpt4')

    gpt35 = AzureChatCompletion(
        api_key = api_key, 
        endpoint = api_endpoint, 
        deployment_name = 'gpt-35-turbo',
        service_id= 'gpt35')

    kernel.add_service(gpt4)
    kernel.add_service(gpt35)

    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    # Retrieve the default inference settings
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings(service_id="gpt35")
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    execution_settings.temperature = 1.0
    # Here you can add additional prompt settings, for example, max_tokens. 

    # Create a chat history object to keep track of the conversation.
    chat_history = ChatHistory()
    chat_history.add_user_message("Finish the following knock-knock joke. Knock, knock. Who's there? Dishes. Dishes who?")

    # Get the chat message content from the chat completion service.
    response = await gpt35.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )
    if response:
        print(f"Assistant:> {response}")

        # Add the chat message to the chat history to keep track of the conversation.
        chat_history.add_message(response)

if __name__ == "__main__":
    asyncio.run(main())