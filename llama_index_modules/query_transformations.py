
# llama_index_modules/query_transformations.py
from llama_index.core.query_engine import TransformQueryEngine
# Corrected Import Path for HyDE
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

def get_hyde_query_engine(index, llm):
    """
    Wraps a standard index with a HyDE query transform engine.
    """
    hyde_transform = HyDEQueryTransform(llm=llm, include_original=True)
    return TransformQueryEngine(index.as_query_engine(), hyde_transform)