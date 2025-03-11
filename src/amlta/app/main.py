import argparse
import logging
import random
from typing import Callable, cast
from uuid import uuid4

import numpy
import pandas as pd
import streamlit as st
import torch

from amlta.formatting.data import create_process_section
from amlta.formatting.markdown import format_as_markdown

st.set_page_config(page_title="ProBas Assistant", layout="wide")
st.title("ProBas Assistant")

from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from streamlit.elements.lib.mutable_status_container import StatusContainer

from amlta.app import config
from amlta.app.agent.core import (
    AgentEvent,
    AgentFinishedEvent,
    AnalyzedFlowsEvent,
    AnalyzingFlowsEvent,
    FetchedFlowsEvent,
    FetchingFlowsEvent,
    ProcessCandidatesFetchedEvent,
    RewritingFlowsQueriesEvent,
    RewritingProcessQueryEvent,
    RewrittenFlowsQueriesEvent,
    RewrittenProcessQueryEvent,
    SelectedProcessEvent,
)
from amlta.probas.processes import ProcessData

logging.basicConfig(level=logging.INFO)

if "messages" not in st.session_state:
    st.session_state.messages = []


UNSET_ARGS = argparse.Namespace(model=None, base_url=None)

chat, side = st.columns([0.65, 0.35])
chat_history = chat.container()
chat_input_container = chat.container()
agent_log_container = side.container()


def handle_event(
    event: AgentEvent,
    process_selection_container: Callable[[], StatusContainer],
    flows_selection_container: Callable[[], StatusContainer],
    flows_analysis_container: Callable[[], StatusContainer],
):
    from amlta.app.agent.graph import cols_to_show, transform_flows_for_analysis

    ev = event.event

    match ev:
        case RewritingProcessQueryEvent():
            process_selection_container().update(label="Rewriting process query...")

        case RewrittenProcessQueryEvent(query=query):
            process_selection_container().update(label="Fetching process candidates...")
            process_selection_container().markdown(
                f"Rewritten process query: `{query.query}`"
            )

        case ProcessCandidatesFetchedEvent(candidates=candidates):
            process_selection_container().update(label="Selecting process...")
            process_selection_container().markdown(
                f"Found {len(candidates)} candidates"
            )
            process_selection_container().markdown(
                "\n".join(
                    f"- `{process.processInformation.dataSetInformation.name.baseName.get()}`"
                    for process in candidates
                )
            )

        case SelectedProcessEvent(process_uuid=process_uuid):
            process = ProcessData.from_uuid(process_uuid)
            name = process.processInformation.dataSetInformation.name.baseName.get()
            process_selection_container().update(
                label=f"Selected process: `{name}`", state="complete", expanded=False
            )
            process_selection_container().markdown(f"Selected process: `{name}`")

            process = ProcessData.from_uuid(process_uuid)
            process_name = (
                process.processInformation.dataSetInformation.name.baseName.get()
            )
            process_data = create_process_section(process, include_flows=False)
            with st.expander(f"Process: `{process_name}`", expanded=False):
                st.code(format_as_markdown(process_data), language="markdown")

        case RewritingFlowsQueriesEvent():
            flows_selection_container().update(label="Rewriting flows queries...")

        case RewrittenFlowsQueriesEvent(rewritten_flows_queries=queries):
            flows_selection_container().update(label="Rewritten flows queries")
            flows_selection_container().markdown(
                "\n".join(f"- `{query.query}`" for query in queries.queries)
            )

        case FetchingFlowsEvent():
            flows_selection_container().update(label="Fetching flows...")

        case FetchedFlowsEvent(flows=flows):
            flows_selection_container().update(label="Fetched flows", state="complete")
            for flow in flows.flows:
                df = pd.DataFrame(flow.filtered)
                df = transform_flows_for_analysis(df)
                flows_selection_container().write(df[cols_to_show])

        case AnalyzingFlowsEvent():
            flows_analysis_container().update(label="Analyzing flows...")

        case AnalyzedFlowsEvent(result=result):
            flows_selection_container().update(expanded=False)

            state = "complete" if not result.exception else "error"

            flows_analysis_container().update(label="Flows analysis", state=state)
            with flows_analysis_container():
                st.code(result.code.code)
                st.markdown("Result")
                if result.result:
                    res_df = pd.DataFrame(result.result)
                    uuid_cols = [
                        col
                        for col in res_df.columns
                        if "uuid" in col or col == "original_index"
                    ]
                    res = res_df.drop(columns=uuid_cols)
                else:
                    res = result.exception

                st.write(res)

        case AgentFinishedEvent(result=result):
            st.write(result["final_answer"])


async def main(args: argparse.Namespace = UNSET_ARGS):
    if args.model:
        config.ollama_model = args.model
    if args.base_url:
        config.ollama_base_url = args.base_url
    if args.use_openai:
        config.use_openai = args.use_openai

    # late import to ensure config was loaded first
    from amlta.app.agent.graph import main as graph

    with chat_history:
        if user_input := chat_input_container.chat_input("Type your message"):
            random.seed(42)
            torch.manual_seed(42)
            numpy.random.seed(42)

            _process_selection_container = None
            _flows_selection_container = None
            _flows_analysis_container = None

            def with_process_selection_container():
                nonlocal _process_selection_container
                if _process_selection_container is None:
                    _process_selection_container = agent_log_container.status(
                        "Process selection", expanded=True
                    )

                return _process_selection_container

            def with_flows_selection_container():
                nonlocal _flows_selection_container
                if _flows_selection_container is None:
                    _flows_selection_container = agent_log_container.status(
                        "Flows selection", expanded=True
                    )

                return _flows_selection_container

            def with_flows_analysis_container():
                nonlocal _flows_analysis_container
                if _flows_analysis_container is None:
                    _flows_analysis_container = agent_log_container.status(
                        "Flows analysis", expanded=True
                    )

                return _flows_analysis_container

            with chat_history.chat_message("human"):
                st.markdown(user_input)

            with chat_history.chat_message("assistant"):
                async for lg_event_type, event in graph.astream(
                    user_input,
                    config={
                        "callbacks": [
                            LoggingCallbackHandler(logging.getLogger("main")),
                        ],
                        "configurable": {"thread_id": uuid4().hex},
                    },
                    stream_mode=["custom", "updates"],
                ):
                    if lg_event_type == "custom":
                        handle_event(
                            cast(AgentEvent, event),
                            process_selection_container=with_process_selection_container,
                            flows_selection_container=with_flows_selection_container,
                            flows_analysis_container=with_flows_analysis_container,
                        )
                    else:
                        pass


async def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llama3.2", help="Ollama model to use"
    )
    parser.add_argument("--base-url", type=str, default=None, help="Ollama base URL")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        default=False,
        help="Use OpenAI instead of Ollama",
    )

    args = parser.parse_args()
    await main(args)


if __name__ == "__main__":
    import asyncio

    asyncio.run(launch())
