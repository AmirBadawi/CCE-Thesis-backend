# General
import os
import asyncio

# Langchain
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.schema import HumanMessage
from langchain_community.llms.azureml_endpoint import (
    AzureMLEndpointApiType,
    LlamaContentFormatter,
)

# Our Modules
import memory as mem
import utils as u


# Main Chat Agent
async def custom_agent(query, memory, conv_id, turbo, index_access):
    try:
        print(index_access)
        response = ""
        user_intent = None
        conversation = mem.memory_to_text(memory) + "Human: " + query
        # print("Running the custom agent...")
        if user_intent is None:
            user_intent = await detect_user_intent_async(conversation)
        print("user intent ==> " + user_intent)
        if "general" in user_intent.strip().lower() or index_access:
            print("Entered General: ")
            response = await others_llm_async(conversation, turbo)
        else:
            print("Entered Intelligencia Knowledge Base: ")
            is_complete = await is_complete_async(query=query)
            print("======is complete?======" + str(is_complete))
            if "incomplete" in str(is_complete).lower():
                query = await condense_chat_async(conversation)
                print("======completed query======" + str(query))
            # uncondensed query
            response = await runCompleteContextRelated_async(query=query, turbo=turbo, conversation=conversation)
        return response
    except Exception:
        raise


async def runCompleteContextRelated_async(query, turbo, conversation):
    print("Entered CompleteContextRelated:")
    try:
        print("Retrieving qa...")
        qa = await get_retrieval_qa_async(query=query, turbo=turbo)
        # qa = await get_qa_chain_async(memory, query)
        print("Running the qa...    ")
        response = await asyncio.wait_for(
            qa.ainvoke(conversation), timeout=int(os.getenv("TIMEOUT", 18))
        )
        print("-"*100)
        print(response["result"])
        return response["result"]
    except Exception:
        raise


async def others_llm_async(conversation, turbo):
    # print("Entered Others:")
    prompt = u.get_prompt("base.txt")
    # history = mem.memory_to_text(memory)
    try:
        if turbo == True:
            print("Turbo GPT ON")
            llm = u.get_chat_turbo_llm()
        else:
            print("Turbo GPT OFF")
            llm = u.get_chat_llm()
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
        print("Running the llm_chain...")
        response = await asyncio.wait_for(
            llm_chain.apredict(query=conversation),
            timeout=int(os.getenv("TIMEOUT", 18)),
        )
    except Exception:
        raise
    return response


async def get_retrieval_qa_async(query, turbo, return_source=False):
    # get the prompt template
    PROMPT = u.get_qabase_prompt()
    # create the chain_type_kwargs
    chain_type_kwargs = {"prompt": PROMPT}
    try:
        # create the RetrievalQA instance and return it (independent from memory)
        if turbo == True:
            print("Turbo GPT ON")
            return RetrievalQA.from_chain_type(
                llm=u.get_chat_turbo_llm(),
                chain_type="stuff",
                retriever=u.get_custom_retriever(query=query),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=return_source,
            )
        else:
            print("Turbo GPT OFF")
            return RetrievalQA.from_chain_type(
                llm=u.get_chat_llm(),
                chain_type="stuff",
                retriever=u.get_custom_retriever(query=query),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=return_source,
            )
    except Exception:
        raise


async def get_qa_chain_async(memory, return_source=False):
    try:
        # return ConversationalRetrievalChain.from_llm(llm=get_chat_llm(), chain_type="stuff", retriever=get_vectorstore().as_retriever(), return_source_documents=return_source, memory=memory)
        return ConversationalRetrievalChain.from_llm(
            llm=u.get_chat_llm(),
            chain_type="stuff",
            retriever=u.get_custom_retriever(),
            memory=memory,
        )
    except Exception:
        raise


async def detect_user_intent_async(conversation):
    prompt = u.get_prompt("intent.txt")
    final_prompt = prompt.format(query=conversation)
    try:
        llm = u.get_chat_llm()
        response = await asyncio.wait_for(
            llm.ainvoke(final_prompt), timeout=int(os.getenv("TIMEOUT", 18))
        )
        print(response.content)
        print("-"*50)
    except Exception:
        raise
    return response.content


async def is_complete_async(query):
    new_query = None
    try:
        prompt_template = u.get_prompt("is_complete.txt")
        question_generator = LLMChain(
            llm=u.get_chat_llm(), prompt=prompt_template, verbose=False
        )
        new_query = question_generator.invoke({"query": query})
        new_query = new_query["text"]
        # print(new_query)
        return new_query
    except Exception as ex:
        # if "content filter" in str(ex).lower():
        #     print(ex)
        #     return query
        # else:
        print(ex)
        raise


async def condense_chat_async(conversation):
    new_query = None
    try:
        # history = mem.memory_to_text(memory)
        prompt_template = u.get_prompt("condense.txt")
        question_generator = LLMChain(
            llm=u.get_chat_llm(), prompt=prompt_template, verbose=False
        )
        new_query = question_generator.invoke({"query": conversation})
        new_query = new_query["text"]
        # print(new_query)
        return new_query
    except Exception as ex:
        print(ex)
        raise


async def history_related_async(memory, query):
    new_query = None
    try:
        history = mem.memory_to_text(memory)
        prompt_template = u.get_prompt_with_memory("history_related.txt")
        question_generator = LLMChain(
            llm=u.get_chat_llm(), prompt=prompt_template, verbose=False
        )
        new_query = question_generator.invoke({"memory": history, "query": query})
        new_query = new_query["text"]
        # print(new_query)
        return new_query
    except Exception as ex:
        print(ex)
        raise


async def condense_query_async(memory, query, related):
    print("Entered condensing: ")
    try:
        new_query = None
        if not related:
            prompt_template = u.get_prompt("condense_unrelated.txt")
            question_generator = LLMChain(
                llm=u.get_chat_llm(), prompt=prompt_template, verbose=False
            )
            new_query = question_generator.invoke({"query": query})
            new_query = new_query["text"]
            print("Condensed new query:", new_query)
            return new_query
        else:
            history = mem.memory_to_text(memory)
            prompt_template = u.get_prompt_with_memory("condense_related.txt")
            question_generator = LLMChain(
                llm=u.get_chat_llm(), prompt=prompt_template, verbose=False
            )
            new_query = question_generator.invoke({"query": query, "memory": history})
            new_query = new_query["text"]
            print("Condensed new query:", new_query)
            return new_query
    except Exception as ex:
        print(ex)
        raise