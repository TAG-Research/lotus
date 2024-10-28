import pickle
from typing import Any

from lotus.models.rm import RM


class ColBERTv2Model(RM):
    """ColBERTv2 Model"""

    def __init__(self, **kwargs):
        self.docs: list[str] | None = None
        self.kwargs: dict[str, Any] = {"doc_maxlen": 300, "nbits": 2, **kwargs}
        self.index_dir: str | None = None

        from colbert import Indexer, Searcher
        from colbert.infra import ColBERTConfig, Run, RunConfig

        self.Indexer = Indexer
        self.Searcher = Searcher
        self.ColBERTConfig = ColBERTConfig
        self.Run = Run
        self.RunConfig = RunConfig

    def index(self, docs: list[str], index_dir: str, **kwargs: dict[str, Any]) -> None:
        kwargs = {**self.kwargs, **kwargs}
        checkpoint = "colbert-ir/colbertv2.0"

        with self.Run().context(self.RunConfig(nranks=1, experiment="lotus")):
            config = self.ColBERTConfig(doc_maxlen=kwargs["doc_maxlen"], nbits=kwargs["nbits"], kmeans_niters=4)
            indexer = self.Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=f"{index_dir}/index", collection=docs, overwrite=True)

        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "wb") as fp:
            pickle.dump(docs, fp)

        self.docs = docs
        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "rb") as fp:
            self.docs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> list:
        raise NotImplementedError("This method is not implemented for ColBERTv2Model")

    def __call__(
        self,
        queries: str | list[str] | list[list[float]],
        k: int,
        **kwargs: dict[str, Any],
    ) -> tuple[list[float], list[int]]:
        if isinstance(queries, str):
            queries = [queries]

        with self.Run().context(self.RunConfig(experiment="lotus")):
            searcher = self.Searcher(index=f"{self.index_dir}/index", collection=self.docs)

        # make queries a dict with keys as query ids
        queries = {i: q for i, q in enumerate(queries)}
        all_results = searcher.search_all(queries, k=k).todict()

        indices = [[result[0] for result in all_results[qid]] for qid in all_results.keys()]
        distances = [[result[2] for result in all_results[qid]] for qid in all_results.keys()]

        return distances, indices
