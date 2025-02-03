import os, asyncio
import semantic_kernel as sk
from semantic_kernel.contents import ChatHistory

# Get keys and configuration from .env file
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.curdir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

# Azure OpenAI Details
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_deployment_name = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_NAME")

# Configure the kernel
kernel = sk.Kernel()

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

gpt4 = AzureChatCompletion(
    api_key = api_key, 
    base_url = api_endpoint, 
    deployment_name = aoai_deployment_name,
    service_id= 'gpt4')

gpt35 = AzureChatCompletion(
    api_key = api_key, 
    base_url = api_endpoint, 
    deployment_name = 'gpt-35',
    service_id= 'gpt35')

kernel.add_service(gpt4)
kernel.add_service(gpt35)


# Main function (needed to make async methods work)

async def main()-> None:

    from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase

    # Retrieve the chat completion service by id
    chat_completion_service = kernel.get_service(service_id="gpt4")

    # Retrieve the default inference settings
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("gpt4")


    chat_history = ChatHistory()
    chat_history.add_user_message("Finish the following knock-knock joke. Knock, knock. Who's there? Dishes. Dishes who?")

    # Get the chat message content from the chat completion service.
    response = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
    )
    if response:
        print(f"Assistant:> {response}")

        # Add the chat message to the chat history to keep track of the conversation.
        chat_history.add_message(response)

if __name__ == "__main__":
    asyncio.run(main())