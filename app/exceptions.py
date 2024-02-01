#imports
# General
import os
import asyncio
from fastapi.datastructures import Default

# Langchain
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain_community.chat_models import AzureChatOpenAI

# Our Modules
import utils as u
from CustomRetriever import CustomRetriever as CustomR


async def generate_timeout_response():
    # print("Entered error generator:")
    try:
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type=os.getenv("OPENAI_TYPE"),
            deployment_name=os.getenv("OPENAI_TURBO_MODEL"),
            verbose=False,
            temperature=0.7,
        )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm.apredict(
                "Generate a user-friendly error message indicating that the API operation has surpassed the specified timeout."
            ),
            timeout=10,
        )

    except asyncio.TimeoutError:
        # print("Exceeded the timeout")
        response = """The operation you requested took longer than expected to complete. Our systems are designed to ensure efficient processing, but occasionally, certain tasks can be time-consuming due to various factors.
                        We apologize for any inconvenience this may have caused."""
    except Exception:
        raise
    return response


async def generate_error_response(ex):
    # print("Entered error generator:")
    try:
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type=os.getenv("OPENAI_TYPE"),
            deployment_name=os.getenv("OPENAI_TURBO_MODEL"),
            verbose=False,
            temperature=0.7,
        )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm.apredict(
                f"As Intelligencia AI Virtual Assistant, generate a brief user-friendly error message for the following:\n{ex}"
            ),
            timeout=10,
        )

    except asyncio.TimeoutError:
        # print("Exceeded the timeout")
        response = """The operation you requested took longer than expected to complete. Our systems are designed to ensure efficient processing, but occasionally, certain tasks can be time-consuming due to various factors.
                        We apologize for any inconvenience this may have caused."""
        return response
    except Exception:
        raise
    return response


async def generate_userfriendly_error_response():
    # print("Entered error generator:")
    try:
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type=os.getenv("OPENAI_TYPE"),
            deployment_name=os.getenv("OPENAI_TURBO_MODEL"),
            verbose=False,
            temperature=0.7,
        )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm.apredict(
                f"As Intelligencia AI Virtual Assistant, generate a brief user friendly error response to try again or contact support."
            ),
            timeout=10,
        )

    except asyncio.TimeoutError:
        # print("Exceeded the timeout")
        response = """The operation you requested took longer than expected to complete. Our systems are designed to ensure efficient processing, but occasionally, certain tasks can be time-consuming due to various factors.
                        We apologize for any inconvenience this may have caused."""
        return response
    except Exception:
        raise
    return response


async def generate_content_filter_error_async(query):
    prompt = u.get_prompt("content_filter_error.txt")
    try:
        llm = u.get_chat_turbo_llm(0.7)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
        )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm_chain.apredict(query=query), timeout=int(os.getenv("TIMEOUT", 18))
        )
        print("Content filter error response ==> " + response)
    except Exception:
        raise
    return response