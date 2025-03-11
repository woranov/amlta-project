import asyncio
import json
import random
import re
import textwrap
from collections import Counter
from collections.abc import Awaitable
from typing import cast

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter
from transformers import (
    TapasForQuestionAnswering,
)

from amlta.app.agent.core import (
    AgentEvent,
    AgentFinishedEvent,
    AgentOutput,
    AnalyzedFlowsEvent,
    AnalyzingFlowsEvent,
    FetchedFlowsEvent,
    FetchingFlowsEvent,
    FilteredFlows,
    FinalFlows,
    FinalFlowsList,
    FlowQueries,
    FlowsQuery,
    FlowValidation,
    PandasCodeOutput,
    ProcessCandidatesFetchedEvent,
    ProcessFlowAnalysisResult,
    RemoveFlowAction,
    RewritingFlowsQueriesEvent,
    RewritingProcessQueryEvent,
    RewrittenFlowsQueriesEvent,
    RewrittenProcessQuery,
    RewrittenProcessQueryEvent,
    SelectedProcess,
    SelectedProcessEvent,
    SelectingProcessEvent,
    load_collections,
)
from amlta.app.llm import get_ollama
from amlta.formatting.data import create_process_section
from amlta.formatting.markdown import format_as_markdown
from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData
from amlta.question_generation.processing import load_batches
from amlta.tapas.model import (
    CustomTapasTokenizer,
)
from amlta.tapas.model import (
    load_tapas_model as _load_tapas_model,
)
from amlta.tapas.model import (
    load_tapas_tokenizer as _load_tapas_tokenizer,
)
from amlta.tapas.retrieve import (
    generate_tapas_chunks,
    retrieve_rows_from_chunk,
)


def inspect_prompt(input: dict):
    print(input)
    return input


@st.cache_resource
def load_tapas_tokenizer() -> CustomTapasTokenizer:
    return _load_tapas_tokenizer()


@st.cache_resource
def load_tapas_model() -> TapasForQuestionAnswering:
    return _load_tapas_model()


base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n{system_prompt}",
        ),
        ("placeholder", "{history}"),
        ("human", "{human_input}"),
    ]
)


rewrite_process_query_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Your task is to rewrite the user question to make it more suitable for searching processes in the
PROBAS database.

## Output ##
The query should completely ignore specifics about flows (i.e., inputs/outputs) and focus only on
the process itself.

## Output format ##
`query`: {RewrittenProcessQuery.model_fields["query"].description}
`justification`: {RewrittenProcessQuery.model_fields["justification"].description}

Return only the rewritten query in `query` and a include your reasoning and thought process in
`justification`.
""".strip()


noop_writer = lambda x: None


@task
async def rewrite_process_query(
    user_question: str, writer: StreamWriter = noop_writer
) -> RewrittenProcessQuery:
    writer(AgentEvent(event=RewritingProcessQueryEvent()))

    llm = get_ollama().with_structured_output(RewrittenProcessQuery)

    chain = base_prompt | inspect_prompt | llm

    res = cast(
        RewrittenProcessQuery,
        await chain.ainvoke(
            {
                "system_prompt": rewrite_process_query_system_prompt,
                "human_input": user_question,
            }
        ),
    )
    writer(AgentEvent(event=RewrittenProcessQueryEvent(query=res)))

    return res


_questions = load_batches()
_q1 = random.choice(_questions)
_q2 = random.choice(_questions)

ex_q1 = _q1["question_replaced_basic"]
ex_r1 = _q1["question"].replace("<", "").replace(">", "")
ex_q2 = _q2["question_replaced_specific"]
ex_r2 = _q2["question"].replace("<", "").replace(">", "")


rewrite_flows_query_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
1. Analyze the user question and categorize it into one of the following categories:
    a) the user asks for one or more specific flows. Multiple flows ARE ALLOWED and MUST NOT be
       splitted
    b) the user asks for one specific class of flows
    c) the user asks for one specific type of flow (elementary flow or waste flow)
2. If and only if the question contains multiple categories, decompose the question into multiple
    queries, one for each category.
3. Given the (decomposed) question, formulate a neutral natural question query -- that is, a query that
    excludes the specific process itself. Furthermore, the query must match the syntax explained in the
    schema field description.

## Output format ##
`queries`: {FlowQueries.model_fields["queries"].description}
`join_type`: {FlowQueries.model_fields["join_type"].description}

Per query:
`justification`: {FlowsQuery.model_fields["justification"].description}
`query`: {FlowsQuery.model_fields["query"].description}

## Example ##
User question: "{ex_q1}"
Rewritten query: "{ex_r1}"

User question: "{ex_q2}"
Rewritten query: "{ex_r2}"
""".strip()


@task
async def rewrite_flows_query(
    user_question: str, writer: StreamWriter = noop_writer
) -> FlowQueries:
    writer(AgentEvent(event=RewritingFlowsQueriesEvent()))

    llm = get_ollama().with_structured_output(
        FlowQueries, method="json_schema", include_raw=True
    )
    chain = base_prompt | llm
    history = []

    tries = 0

    while tries < 3:
        resp = cast(
            dict,
            await chain.ainvoke(
                {
                    "system_prompt": rewrite_flows_query_system_prompt,
                    "history": history,
                    "human_input": user_question,
                }
            ),
        )

        if resp["parsing_error"]:
            print(resp["parsing_error"])
            tries += 1

            history.append(
                SystemMessage(
                    content=f"Error: {resp['parsing_error']}\nPlease fix your mistakes."
                )
            )
        else:
            res = cast(FlowQueries, resp["parsed"])
            writer(
                AgentEvent(
                    event=RewrittenFlowsQueriesEvent(rewritten_flows_queries=res)
                )
            )
            return res
    else:
        raise RuntimeError("Failed to rewrite flows query after 3 attempts")


select_process_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Your task is to select the best fitting process from the list of candidates.
You will receive a list of candidate processes and you must select the best fitting one.

## Output format ##
`justification`: {SelectedProcess.model_fields["justification"].description}
`index`: {SelectedProcess.model_fields["index"].description}
""".strip()


@task
async def select_process(
    candidates: list[ProcessData],
    user_question: str,
    writer: StreamWriter = noop_writer,
) -> SelectedProcess:
    writer(AgentEvent(event=SelectingProcessEvent()))

    llm = get_ollama().with_structured_output(SelectedProcess)
    chain = base_prompt | llm

    def _format_candidate(index: int, process: ProcessData) -> str:
        data = create_process_section(process, include_flows=False)
        keep_keys = ["Name", "Year", "Geography", "Class", "Main Output"]
        data = {k: v for k, v in data.items() if k in keep_keys}

        return "<candidate index={index}>\n{data}\n</candidate>".format(
            index=index, data=json.dumps(data, indent=2)
        )

    human_template = (
        "<question>{question}</question>\n<processes>\n{candidates}\n</processes>"
    )

    res = cast(
        SelectedProcess,
        await chain.ainvoke(
            {
                "system_prompt": select_process_system_prompt,
                "human_input": human_template.format(
                    question=user_question,
                    candidates="\n\n".join(
                        _format_candidate(idx, c) for idx, c in enumerate(candidates)
                    ),
                ),
            }
        ),
    )
    writer(
        AgentEvent(
            event=SelectedProcessEvent(
                process=res,
                process_uuid=candidates[
                    res.index
                ].processInformation.dataSetInformation.UUID,
            )
        )
    )
    return res


filter_flows_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Given a list of flows, your task is verify if the flows match the user question.

## Output format ##
`justification`: {FlowValidation.model_fields["justification"].description}
`removals`: {FlowValidation.model_fields["removals"].description}

Per removal:
`justification`: {RemoveFlowAction.model_fields["justification"].description}
`index`: {RemoveFlowAction.model_fields["index"].description}
"""


def _format_flow(index: int, flow: dict) -> str:
    return "<flow index={index}>\n{data}\n</flow>".format(
        index=index, data=json.dumps(flow, indent=2)
    )


@task
async def fetch_flows(
    flows_df: pd.DataFrame,
    threshold: float,
    query: FlowsQuery,
    writer: StreamWriter = noop_writer,
) -> FilteredFlows:
    model = load_tapas_model()
    tokenizer = load_tapas_tokenizer()

    flows_df = flows_df.reset_index().rename(columns={"index": "original_index"})

    retrieval_tasks = []
    for chunk in generate_tapas_chunks(flows_df):
        retrieval_tasks.append(
            asyncio.to_thread(
                retrieve_rows_from_chunk,
                chunk,
                query=query.query,
                model=model,
                tokenizer=tokenizer,
                threshold=threshold,
            )
        )
    retrieval_results = await asyncio.gather(*retrieval_tasks)

    candidate_indices = set()
    agg_counter = Counter()
    for indices, aggregation in retrieval_results:
        candidate_indices.update(indices)
        agg_counter[aggregation] += 1

    aggregation = agg_counter.most_common(1)[0][0]

    merged_candidates = flows_df.loc[flows_df["original_index"].isin(candidate_indices)]

    candidate_chunks = [
        merged_candidates.iloc[i : i + 15].reset_index(drop=True)
        for i in range(0, len(merged_candidates), 15)
    ]

    filter_tasks = [
        filter_flows_llm_chunk(chunk, query.query, aggregation)
        for chunk in candidate_chunks
    ]
    filter_results = cast(list[FilteredFlows], await asyncio.gather(*filter_tasks))

    final_indices = []
    for res in filter_results:
        final_indices.extend(res.flow_indices)

    return FilteredFlows(flow_indices=final_indices, aggregation=aggregation)


async def filter_flows_llm_chunk(
    chunk: pd.DataFrame,
    query: str,
    aggregation: str,
) -> FilteredFlows:
    if chunk.empty:
        return FilteredFlows(flow_indices=[], aggregation=aggregation)

    llm = get_ollama().with_structured_output(FlowValidation)
    chain = base_prompt | llm

    human_template = "<question>{question}</question>\n<flows>\n{flows}\n</flows>"

    chunk = transform_flows_for_analysis(chunk)
    formatted_flows = "\n\n".join(
        _format_flow(idx, flow)
        for idx, flow in enumerate(chunk[cols_to_show].to_dict(orient="records"))
    )
    res = cast(
        FlowValidation,
        await chain.ainvoke(
            {
                "system_prompt": filter_flows_system_prompt,
                "human_input": human_template.format(
                    question=query, flows=formatted_flows
                ),
            }
        ),
    )
    removals_indices = [rem.index for rem in res.removals]
    filtered_chunk = chunk.drop(
        chunk.iloc[removals_indices].index.tolist(), axis=0, inplace=False
    )

    return FilteredFlows(
        flow_indices=filtered_chunk["original_index"].tolist(), aggregation=aggregation
    )


_code_fn_template = """
def analyze_results(dataframes) -> pd.DataFrame:
{code}
"""


_code_template = """
{fn}

try:
    result = analyze_results(dataframes)
except Exception as e:
    exception = e
    result = None
else:
    exception = None
"""


def _fix_code(code: str):
    code = code.lstrip("\n\r")
    if not code.startswith("    ") and not "def analyze_results" in code:
        code = textwrap.indent(code, "    ")
    elif "def analyze_results" in code:
        code = re.sub(r"def analyze_results.*?\n", "", code, count=1)

    return _code_fn_template.format(code=code)


def _format_code(code: str):
    return _code_template.format(fn=_fix_code(code))


def _interpret_python(locals: dict[str, object], code: str):
    print("RUNNING")
    print(code)

    env = {"__builtins__": __builtins__, **locals}

    exec(code, env)

    result = env["result"]
    exception = env["exception"]

    if isinstance(result, pd.Series):
        result = result.to_frame().T

    return result.to_dict(orient="records"), exception


analyze_results_system_prompt = rf"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Given one or more dataframes containing flows (i.e., inputs/outputs) of a process, your task is to
generate python code that analyzes the dataframes and returns a dataframe answering the user
question.

## Output format ##
`justification`: {PandasCodeOutput.model_fields["justification"].description}
`code`: {PandasCodeOutput.model_fields["code"].description}

## Example ##
```python
import pandas as pd

process_name = r"Metall\Fe-roh-DE-2005"

dataframes = [
    # df 0 contains data answering 'What are the output amounts of lead to water of the process?'
    df0,
    # df 1 contains data answering 'What are the output amounts of mercury to water of the process?'
    df1
]

# question: Does iron production emit more mercury to water or lead to water?
def analyze_results(dataframes) -> pd.DataFrame:
    # your code
    df0 = dataframes[0]
    df1 = dataframes[1]
    return pd.concat([df0, df1]).groupby(["name", "unit"]).sum().sort_values("amount", ascending=False)
```

For reference:
```python
df0.head(1)
```
```pycon
  direction            type class name amount unit unit_property_name
0     input Elementary Flow   ... lead   0.01   kg               Mass
```

Your response:
{{ "code":  "df0 = ...", "justification": "The code compares the total output amounts of lead and mercury to water by combining the dataframes and ordering the result." }}
"""

analyze_results_human_template = """
```python
import pandas as pd

process_name = r"{process_name}"

dataframes = [
{dataframes}
]

# question: {question}
def analyze_results(dataframes):
    # your code

```

{df_heads}
""".strip()


df_head_template = """
```python
df{index}.head(5)
```
```pycon
{df_head}
```
"""

# direction       type  ...                     name       amount unit  unit_property_name
# 0   input Waste Flow  ...  secondary raw materials 3.960000e+00   MJ Net calorific value


cols_to_show = [
    "direction",
    "type",
    "class",
    "name",
    "amount",
    "unit",
    "unit_property_name",
]


def transform_flows_for_analysis(flows_df: pd.DataFrame):
    df = flows_df.rename(
        columns={
            "exchange_direction": "direction",
            "exchange_type_of_flow": "type",
            "exchange_classification_hierarchy": "class",
            "flow_description": "name",
            "exchange_resulting_amount": "amount",
            "flow_property_unit": "unit",
            "flow_property_name": "unit_property_name",
        }
    ).assign(
        direction=lambda df: df["direction"].str.lower(),
    )
    other_cols = df.columns.difference(cols_to_show).to_list()
    return df[cols_to_show + other_cols]


@task
async def analyze_results(
    original_question: str,
    rewritten_process_query: RewrittenProcessQuery,
    selected_process: SelectedProcess,
    selected_process_uuid: str,
    rewritten_flows_queries: FlowQueries,
    final_flows: FinalFlowsList,
    tries_left: int = 3,
    history: list[BaseMessage] | None = None,
    writer: StreamWriter = noop_writer,
) -> ProcessFlowAnalysisResult:
    process = ProcessData.from_uuid(selected_process_uuid)
    process_name = process.processInformation.dataSetInformation.name.baseName.get()

    dfs: list[pd.DataFrame] = []
    formatted_df_list = []

    for i, final_flow in enumerate(final_flows.flows):
        flows_df = pd.DataFrame(final_flow.filtered)
        flows_df = transform_flows_for_analysis(flows_df)
        dfs.append(flows_df)

        formatted_df_list.append((f"    # {final_flow.query.query}\n    df{i},"))

    formatted_dfs = "\n".join(formatted_df_list)

    with pd.option_context("display.max_columns", None):
        df_heads = "\n\n".join(
            df_head_template.format(index=i, df_head=repr(df[cols_to_show].head(5)))
            for i, df in enumerate(dfs)
        )

    human_input = analyze_results_human_template.format(
        process_name=process_name,
        dataframes=formatted_dfs,
        question=original_question,
        df_heads=df_heads,
    )

    llm = get_ollama().with_structured_output(PandasCodeOutput, method="json_schema")
    chain = base_prompt | llm

    res = cast(
        PandasCodeOutput,
        await chain.ainvoke(
            {
                "system_prompt": analyze_results_system_prompt,
                "history": history or [],
                "human_input": human_input,
            }
        ),
    )

    fn_locals = {
        "pd": pd,
        "dataframes": dfs,
        "process_name": process_name,
        **{f"df{i}": df for i, df in enumerate(dfs)},
    }

    result, exception = await asyncio.to_thread(
        _interpret_python, locals=fn_locals, code=_format_code(res.code)
    )

    res.code = _fix_code(res.code)

    if exception is not None and tries_left > 0:
        tries_left -= 1
        feedback = f"Error: {exception}\nPlease fix your mistakes."
        history = [
            AIMessage(content=res.model_dump_json()),
            SystemMessage(content=feedback),
        ]
        return cast(
            ProcessFlowAnalysisResult,
            await analyze_results(
                original_question=original_question,
                rewritten_process_query=rewritten_process_query,
                selected_process=selected_process,
                selected_process_uuid=selected_process_uuid,
                rewritten_flows_queries=rewritten_flows_queries,
                final_flows=final_flows,
                tries_left=tries_left,
                history=history,
            ),
        )

    return ProcessFlowAnalysisResult(
        code=res,
        result=result,
        exception=exception,
    )


final_answer_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Your task is to answer the user question based on the provided context and generated intermediate
results.

## Output format ##
Respond in text format, addressing the user's question directly. Convey if the question could be
answered appropriately or not. Include any possible limitations or uncertainties in your answer.

Use well formatted markdown as well as professional grammar and spelling.
""".strip()


final_answer_human_template = """
{question}

<context>
<!-- potentially useful glossary terms -->
{glossary}

<!-- the process that was selected -->
{process_context}

<!-- the flows that were queried -->
{flows_context}

<!-- the pandas analysis result of the flows -->
{analysis_result}
</context>

<question>
{question}
</question>
""".strip()


@task
async def get_final_answer(
    original_question: str,
    process: ProcessData,
    flows_queries: FlowQueries,
    flows_results: FinalFlowsList,
    analysis_result: ProcessFlowAnalysisResult,
    writer: StreamWriter = noop_writer,
):
    llm = get_ollama()
    chain = base_prompt | llm

    retriever = load_collections().glossary.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15}
    )

    process_data = create_process_section(process, include_flows=False)
    process_markdown = format_as_markdown(process_data)
    process_context = "<process>{}</process>".format(
        textwrap.indent(process_markdown, "    ")
    )

    docs = retriever.invoke(original_question + " " + process_markdown)
    glossary_terms = textwrap.indent(
        "\n\n".join(f"<term>{doc.page_content}</term>" for doc in docs),
        "    ",
    )
    glossary_context = "<glossary>\n{terms}\n</glossary>".format(
        terms=glossary_terms,
    )

    flows_contexts = []

    for i, flows in enumerate(flows_results.flows):
        df_name = f"df{i}"

        flows_context_query = (
            f'<flow-query name="{df_name}">{flows.query.query}</flow-query>'
        )

        flows_context_result = "\n".join(
            f'<flow-result name="{df_name}">{json.dumps(flow_data, indent=2)}</flow-result>'
            for flow_data in flows.filtered[:5]
        ) + (
            "\n<!-- ... {} more flows omitted -->".format(len(flows.filtered) - 5)
            if len(flows.filtered) > 5
            else ""
        )

        flows_contexts.append(
            "{query}\n{result}".format(
                query=textwrap.indent(flows_context_query, "    "),
                result=textwrap.indent(flows_context_result, "    "),
            )
        )

    flows_context = (
        "<intermediary-results>\n{flows_results}\n</intermediary-results>".format(
            flows_results="\n".join(flows_contexts)
        )
    )

    analysis_result_dump = "<result>{}</result>".format(
        textwrap.indent(
            json.dumps(analysis_result.result, indent=2),
            "    ",
        )
    )
    analysis_result_context = "<analysis-result>{}</analysis-result>".format(
        textwrap.indent(analysis_result_dump, "    ")
    )
    human_input = final_answer_human_template.format(
        question=original_question,
        glossary=glossary_context,
        process_context=process_context,
        flows_context=flows_context,
        analysis_result=analysis_result_context,
    )

    return cast(
        str,
        (
            await chain.ainvoke(
                {
                    "system_prompt": final_answer_system_prompt,
                    "human_input": human_input,
                }
            )
        ).content,
    )


@entrypoint(checkpointer=MemorySaver())
async def main(user_question: str, writer: StreamWriter) -> AgentOutput:
    rewritten_process_query_fut = cast(
        Awaitable[RewrittenProcessQuery], rewrite_process_query(user_question)
    )

    rewritten_flows_query_fut = cast(
        Awaitable[FlowQueries], rewrite_flows_query(user_question)
    )

    rewritten_process_query_resp = await rewritten_process_query_fut
    rewritten_process_query = rewritten_process_query_resp.query

    collections = load_collections()
    # candidate_processes_docs = collections.processes.similarity_search(
    #     rewritten_process_query, k=25
    # )
    # ranked = collections.reranker.rank(
    #     rewritten_process_query,
    #     [doc.page_content for doc in candidate_processes_docs],
    #     top_k=10,
    #     num_workers=0,
    # )
    # reranked_docs = [
    #     candidate_processes_docs[res["corpus_id"]]  # type: ignore
    #     for res in ranked
    # ]
    reranked_docs = collections.processes.similarity_search(
        rewritten_process_query, k=10
    )

    candidate_processes: list[ProcessData] = [
        ProcessData.from_uuid(doc.metadata["uuid"]) for doc in reranked_docs
    ]
    writer(
        AgentEvent(event=ProcessCandidatesFetchedEvent(candidates=candidate_processes))
    )

    selected_process_resp = cast(
        SelectedProcess,
        await select_process(candidate_processes, rewritten_process_query),
    )

    selected_process = candidate_processes[selected_process_resp.index]
    selected_process_uuid = selected_process.processInformation.dataSetInformation.UUID

    rewritten_flows_queries = await rewritten_flows_query_fut

    flows_df = extract_process_flows(selected_process)

    writer(AgentEvent(event=FetchingFlowsEvent()))
    final_flows: FinalFlowsList = FinalFlowsList(
        join_type=rewritten_flows_queries.join_type, flows=[]
    )

    futures = []
    for query in rewritten_flows_queries.queries:
        futures.append(
            fetch_flows(
                flows_df,
                threshold=0.05,
                query=query,
            )
        )

    flows_df = flows_df.reset_index().rename(columns={"index": "original_index"})

    for flows_query, filtered_flows in zip(
        rewritten_flows_queries.queries, await asyncio.gather(*futures)
    ):
        filtered_flows = cast(FilteredFlows, filtered_flows)

        final_flows.flows.append(
            FinalFlows(
                query=flows_query,
                filtered=flows_df.loc[
                    flows_df["original_index"].isin(filtered_flows.flow_indices)
                ].to_dict(orient="records"),
            )
        )

    writer(AgentEvent(event=FetchedFlowsEvent(flows=final_flows)))
    writer(AgentEvent(event=AnalyzingFlowsEvent()))

    analysis_result = cast(
        ProcessFlowAnalysisResult,
        await analyze_results(
            original_question=user_question,
            rewritten_process_query=rewritten_process_query_resp,
            selected_process=selected_process_resp,
            selected_process_uuid=selected_process_uuid,
            rewritten_flows_queries=rewritten_flows_queries,
            final_flows=final_flows,
        ),
    )

    writer(AgentEvent(event=AnalyzedFlowsEvent(result=analysis_result)))

    answer = cast(
        str,
        await get_final_answer(
            original_question=user_question,
            process=selected_process,
            flows_queries=rewritten_flows_queries,
            flows_results=final_flows,
            analysis_result=analysis_result,
        ),
    )

    res: AgentOutput = {
        "initial_question": user_question,
        "rewritten_process_query": rewritten_process_query_resp,
        "selected_process": selected_process_resp,
        "selected_process_uuid": selected_process_uuid,
        "rewritten_flows_queries": rewritten_flows_queries,
        "final_flows": final_flows,
        "analysis_result": analysis_result,
        "final_answer": answer,
    }

    writer(AgentEvent(event=AgentFinishedEvent(type="agent_finished", result=res)))

    return res
