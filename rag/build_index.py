"""사기 사례 RAG 인덱스 구축.

SEC 제재 결정문 등 사기 사례 텍스트를 임베딩하여 ChromaDB에 저장.
BAAI/bge-m3 임베딩 모델 사용 (~2GB VRAM).
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = BASE_DIR / "data" / "cases"
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_db"


def load_cases(cases_dir: Path) -> list[dict]:
    """사기 사례 JSON 파일들을 로드."""
    cases = []

    for subdir in sorted(cases_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            if json_file.name == "crawl_summary.json":
                continue
            try:
                with open(json_file, encoding="utf-8") as f:
                    case = json.load(f)
                if case.get("content"):
                    cases.append(case)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"  Skip {json_file}: {e}")

    return cases


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """텍스트를 청크로 분할."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 50:  # 너무 짧은 청크 제외
            chunks.append(chunk)

    return chunks


def build_index(
    cases_dir: str,
    embedding_model: str = "BAAI/bge-m3",
    output_dir: str | None = None,
    chunk_size: int = 500,
) -> None:
    """ChromaDB 인덱스 구축."""
    cases_path = Path(cases_dir)
    db_path = Path(output_dir) if output_dir else DEFAULT_DB_DIR

    # 사례 로드
    print("[1/3] Loading fraud cases...")
    cases = load_cases(cases_path)
    print(f"  Loaded {len(cases)} cases with content")

    if not cases:
        print("No cases found. Run crawl_sec.py first.")
        return

    # 임베딩 모델 로드
    print(f"\n[2/3] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ChromaDB 컬렉션 생성
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("fraud_cases")
    except ValueError:
        pass
    collection = client.create_collection(
        name="fraud_cases",
        metadata={"description": "Financial fraud case database for RAG"},
    )

    # 청크 생성 + 임베딩
    print(f"\n[3/3] Chunking and embedding...")
    all_docs = []
    all_ids = []
    all_metadatas = []
    all_embeddings = []

    for case in cases:
        chunks = chunk_text(case["content"], chunk_size=chunk_size)
        case_meta = case.get("metadata", {})

        for j, chunk in enumerate(chunks):
            doc_id = f"{case['id']}_chunk_{j}"
            metadata = {
                "case_id": case.get("id", "unknown"),
                "title": case.get("title", "")[:200],
                "source": case.get("source", ""),
                "year": str(case.get("year", "")),
                "fraud_types": ", ".join(case_meta.get("fraud_types", [])),
                "chunk_index": j,
                "total_chunks": len(chunks),
            }

            all_docs.append(chunk)
            all_ids.append(doc_id)
            all_metadatas.append(metadata)

    print(f"  Total chunks: {len(all_docs)}")

    # 배치 임베딩
    batch_size = 32
    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i : i + batch_size]
        embeddings = model.encode(batch_docs, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        if (i + batch_size) % 100 == 0:
            print(f"  Embedded {min(i + batch_size, len(all_docs))}/{len(all_docs)} chunks")

    # ChromaDB에 추가
    # ChromaDB는 배치 크기 제한이 있으므로 분할
    add_batch = 500
    for i in range(0, len(all_docs), add_batch):
        end = min(i + add_batch, len(all_docs))
        collection.add(
            documents=all_docs[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end],
        )

    print(f"\nIndex built: {collection.count()} chunks in {db_path}")

    # 인덱스 통계 저장
    stats = {
        "total_cases": len(cases),
        "total_chunks": len(all_docs),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "db_path": str(db_path),
    }
    stats_path = db_path / "index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build fraud case RAG index")
    parser.add_argument("--cases_dir", type=str, default=str(CASES_DIR))
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=500)
    args = parser.parse_args()

    build_index(args.cases_dir, args.embedding_model, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
