"""사기 사례 RAG 검색.

ChromaDB에서 유사한 사기 사례를 검색하여 LLM 컨텍스트로 제공.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_db"


class FraudCaseSearcher:
    """사기 사례 검색 엔진."""

    def __init__(
        self,
        db_dir: str | None = None,
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.db_path = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        self.model = None
        self.collection = None
        self.embedding_model_name = embedding_model

    def load(self) -> None:
        """모델과 DB 로드."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {self.db_path}. Run build_index.py first."
            )

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)

        client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = client.get_collection("fraud_cases")
        print(f"Loaded {self.collection.count()} chunks from {self.db_path}")

    def search(
        self,
        query: str,
        top_k: int = 3,
        fraud_type_filter: str | None = None,
    ) -> list[dict]:
        """유사 사기 사례 검색."""
        if self.model is None:
            self.load()

        query_embedding = self.model.encode(query).tolist()

        where_filter = None
        if fraud_type_filter:
            where_filter = {"fraud_types": {"$contains": fraud_type_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # 같은 케이스 중복 제거를 위해 더 많이 검색
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 같은 case_id 중복 제거 (가장 관련 높은 청크만)
        seen_cases = set()
        unique_results = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            case_id = meta.get("case_id", "unknown")
            if case_id in seen_cases:
                continue
            seen_cases.add(case_id)

            similarity = 1 - dist  # cosine distance → similarity
            unique_results.append({
                "id": case_id,
                "title": meta.get("title", "Unknown"),
                "fraud_type": meta.get("fraud_types", "Unknown"),
                "source": meta.get("source", ""),
                "year": meta.get("year", ""),
                "summary": doc[:300],  # 청크의 처음 300자를 요약으로
                "similarity": similarity,
            })

            if len(unique_results) >= top_k:
                break

        return unique_results

    def search_for_transaction(
        self,
        transaction_text: str,
        fraud_score: float,
        top_k: int = 3,
    ) -> list[dict]:
        """거래 텍스트로 유사 사례 검색 (검색 쿼리 최적화)."""
        # 거래 정보를 사기 사례 검색에 적합한 쿼리로 변환
        query = f"Financial fraud case involving: {transaction_text}"
        if fraud_score > 0.9:
            query += " High confidence fraud detection."

        return self.search(query, top_k=top_k)
