"""TF-IDF 기반 사기 사례 검색 (오프라인 fallback).

sentence-transformers 모델 다운로드가 불가능할 때 사용.
scikit-learn TF-IDF + cosine similarity로 유사 사례 검색.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = BASE_DIR / "data" / "cases"
INDEX_DIR = Path(__file__).resolve().parent / "tfidf_index"


class TfidfFraudSearcher:
    """TF-IDF 기반 사기 사례 검색."""

    def __init__(self, index_dir: str | None = None):
        self.index_path = Path(index_dir) if index_dir else INDEX_DIR
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.metadata = []

    def build(self, cases_dir: str | None = None) -> None:
        """인덱스 구축."""
        cases_path = Path(cases_dir) if cases_dir else CASES_DIR
        print("[1/2] Loading fraud cases...")

        for subdir in sorted(cases_path.iterdir()):
            if not subdir.is_dir():
                continue
            for json_file in sorted(subdir.glob("*.json")):
                if json_file.name == "crawl_summary.json":
                    continue
                try:
                    with open(json_file, encoding="utf-8") as f:
                        case = json.load(f)
                    if case.get("content"):
                        self.documents.append(case["content"])
                        self.metadata.append({
                            "id": case.get("id", "unknown"),
                            "title": case.get("title", "")[:200],
                            "source": case.get("source", ""),
                            "year": str(case.get("year", "")),
                            "fraud_types": ", ".join(
                                case.get("metadata", {}).get("fraud_types", [])
                            ),
                            "amounts": case.get("metadata", {}).get("amounts_mentioned", []),
                            "regulations": case.get("metadata", {}).get("regulations", []),
                        })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        print(f"  Loaded {len(self.documents)} cases")

        print("[2/2] Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # 저장
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.index_path / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(self.index_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.index_path / "documents.json", "w") as f:
            json.dump(self.documents, f, ensure_ascii=False)

        print(f"  Index saved to {self.index_path}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def load(self) -> None:
        """저장된 인덱스 로드."""
        with open(self.index_path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.index_path / "tfidf_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self.index_path / "metadata.json") as f:
            self.metadata = json.load(f)
        with open(self.index_path / "documents.json") as f:
            self.documents = json.load(f)
        print(f"Loaded TF-IDF index: {len(self.documents)} cases")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """유사 사례 검색."""
        if self.vectorizer is None:
            self.load()

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] < 0.01:
                continue
            meta = self.metadata[idx]
            results.append({
                "id": meta["id"],
                "title": meta["title"],
                "fraud_type": meta["fraud_types"],
                "source": meta["source"],
                "year": meta["year"],
                "summary": self.documents[idx][:300],
                "similarity": float(similarities[idx]),
                "amounts_mentioned": meta.get("amounts", []),
                "regulations": meta.get("regulations", []),
            })

        return results

    def search_for_transaction(
        self,
        transaction_text: str,
        fraud_score: float,
        top_k: int = 3,
    ) -> list[dict]:
        """거래 텍스트 기반 검색."""
        query = f"Financial fraud: {transaction_text}"
        return self.search(query, top_k=top_k)


if __name__ == "__main__":
    searcher = TfidfFraudSearcher()
    searcher.build()

    # 테스트 검색
    print("\n=== Test Search ===")
    results = searcher.search("large cash deposits structured below reporting threshold")
    for r in results:
        print(f"\n[{r['id']}] {r['title']}")
        print(f"  Type: {r['fraud_type']}, Similarity: {r['similarity']:.3f}")
