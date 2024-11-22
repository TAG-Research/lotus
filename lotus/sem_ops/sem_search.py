from typing import Any
import pandas as pd
import lotus
from lotus.types import RerankerOutput, RMOutput


@pd.api.extensions.register_dataframe_accessor("sem_search")
class SemSearchDataframe:
    """DataFrame accessor for semantic search."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        col_name: str,
        query: str,
        K: int | None = None,
        n_rerank: int | None = None,
        return_scores: bool = False,
        suffix: str = "_sim_score",
        chunk_size: int = -1  # New parameter for setting chunk size
    ) -> pd.DataFrame:
        """
        Perform semantic search on the DataFrame.

        Args:
            col_name (str): The column name to search on.
            query (str): The query string.
            K (int | None): The number of documents to retrieve.
            n_rerank (int | None): The number of documents to rerank.
            return_scores (bool): Whether to return the similarity scores.
            suffix (str): The suffix to append to the new column containing the similarity scores.
            chunk_size (int): The size of chunks for memory optimization (-1 to disable).

        Returns:
            pd.DataFrame: The DataFrame with the search results.
        """
        assert not (K is None and n_rerank is None), "K or n_rerank must be provided"

        if K is not None:
            # get retriever model and index
            rm = lotus.settings.rm
            col_index_dir = self._obj.attrs["index_dirs"][col_name]
            if rm.index_dir != col_index_dir:
                rm.load_index(col_index_dir)
            assert rm.index_dir == col_index_dir

            df_idxs = self._obj.index
            K = min(K, len(df_idxs))

            search_K = K
            while True:
                if chunk_size > 0:
                    # Process in specified chunk sizes to reduce memory usage
                    scores, doc_idxs = [], []
                    for start in range(0, search_K, chunk_size):
                        end = min(start + chunk_size, search_K)
                        rm_output: RMOutput = rm(query, end - start)
                        chunk_scores = rm_output.distances[0]
                        chunk_doc_idxs = rm_output.indices[0]                        
                        scores.extend(chunk_scores[0])
                        doc_idxs.extend(chunk_doc_idxs[0])
                else:
                    rm_output: RMOutput = rm(query, search_K)
                    doc_idxs = rm_output.indices[0]
                    scores = rm_output.distances[0]

                assert len(doc_idxs) == len(scores)

                postfiltered_doc_idxs = []
                postfiltered_scores = []
                for idx, score in zip(doc_idxs, scores):
                    if idx in df_idxs:
                        postfiltered_doc_idxs.append(idx)
                        postfiltered_scores.append(score)

                postfiltered_doc_idxs = postfiltered_doc_idxs[:K]
                postfiltered_scores = postfiltered_scores[:K]
                if len(postfiltered_doc_idxs) == K:
                    break
                search_K = search_K * 2

            new_df = self._obj.loc[postfiltered_doc_idxs]
            new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)

            if return_scores:
                new_df["vec_scores" + suffix] = postfiltered_scores
        else:
            new_df = self._obj

        if n_rerank is not None:
            docs = new_df[col_name].tolist()
            reranked_output: RerankerOutput = lotus.settings.reranker(query, docs, n_rerank)
            reranked_idxs = reranked_output.indices
            new_df = new_df.iloc[reranked_idxs]

        return new_df
