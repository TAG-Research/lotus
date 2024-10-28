from typing import Any

import pandas as pd

import lotus


@pd.api.extensions.register_dataframe_accessor("sem_sim_join")
class SemSimJoinDataframe:
    """DataFrame accessor for semantic similarity join."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
    ) -> pd.DataFrame:
        """
        Perform semantic similarity join on the DataFrame.

        Args:
            other (pd.DataFrame): The other DataFrame to join with.
            left_on (str): The column name to join on in the left DataFrame.
            right_on (str): The column name to join on in the right DataFrame.
            K (int): The number of nearest neighbors to search for.
            lsuffix (str): The suffix to append to the left DataFrame.
            rsuffix (str): The suffix to append to the right DataFrame.
            score_suffix (str): The suffix to append to the similarity score column.
        """

        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        # get rmodel and index
        rm = lotus.settings.rm

        # load query embeddings from index if they exist
        if left_on in self._obj.attrs.get("index_dirs", []):
            query_index_dir = self._obj.attrs["index_dirs"][left_on]
            if rm.index_dir != query_index_dir:
                rm.load_index(query_index_dir)
            assert rm.index_dir == query_index_dir
            try:
                queries = rm.get_vectors_from_index(query_index_dir, self._obj.index)
            except NotImplementedError:
                queries = self._obj[left_on].tolist()
        else:
            queries = self._obj[left_on].tolist()

        # load index to search over
        try:
            col_index_dir = other.attrs["index_dirs"][right_on]
        except KeyError:
            raise ValueError(f"Index directory for column {right_on} not found in DataFrame")
        if rm.index_dir != col_index_dir:
            rm.load_index(col_index_dir)
        assert rm.index_dir == col_index_dir

        distances, indices = rm(queries, K)

        other_index_set = set(other.index)
        join_results = []

        # post filter
        for q_idx, res_ids in enumerate(indices):
            for i, res_id in enumerate(res_ids):
                if res_id != -1 and res_id in other_index_set:
                    join_results.append((self._obj.index[q_idx], res_id, distances[q_idx][i]))

        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = df1.index
        df2["_right_id"] = df2.index
        temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", "_scores" + score_suffix])
        joined_df = (
            df1.join(
                temp_df.set_index("_left_id"),
                how="right",
                on="_left_id",
            )
            .join(
                df2.set_index("_right_id"),
                how="left",
                on="_right_id",
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            )
            .drop(columns=["_left_id", "_right_id"])
        )

        return joined_df
