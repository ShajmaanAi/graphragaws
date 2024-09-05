import os
import pandas as pd
import tiktoken
import asyncio
from rich import print
from graphrag.query.indexer_adapters import read_indexer_entities,read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

#setup llm

api_key=os.environ["GRAPHRAG_API_KEY"]
llm_model ="mistral:instruct"
llm=ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20
)
token_encoder= tiktoken.get_encoding("Cl100k_base")

#label content

INPUT_DIR="input"
COMMUNITY_REPORT_TABLE ="create_final_community_reports"
ENTITY_TABLE="create_final_nodes"
ENTITY_EMBEDDING_TABLE ="create fial entities"


COMMUNITY_LEVEL =0
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.paraquet")
report_df=pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.paraquet")
entity_embedding_df=pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.paraquet")

reports=read_indexer_reports(report_df,entity_df,COMMUNITY_LEVEL)
entities=read_indexer_entities(report_df,entity_embedding_df,COMMUNITY_LEVEL)
print(f"Report Records : {len(report_df)}")
report_df.head()


context_builder=GlobalCommunityContext(
    community_reports=reports,
    entities=entities,
    token_encoder=token_encoder
)

context_builder_params={"use_community_summary":False,"shuffle_data":True,"include_community_rank":True,}
map_llm_params={"max_tokens":1000,"temperature":0.0,"response_format":{"type":"json_object"},}
reduce_llm_params={"max_tokens":2000,"temperature":0.0,}

search_engine=GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12_000,
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,
    json_mode=True,
    context_builder_params=context_builder_params,
    concurrent_coroutines=10,
)

async def main(query: str):
    result= await search_engine.asearch(query)
    return result

if __name__=="__main__":
    query="VLM future work"
    result=asyncio.run(main(query))
    print(result.response)
    print(result.context_data["reports"])
    print(f"LLM calls{result.llm_calls}.LLM tokens: {result.prompt_tokens}")


