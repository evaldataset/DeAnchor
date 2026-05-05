"""SEC Enforcement Actions 수집 스크립트.

SEC EDGAR에서 금융 사기 관련 제재 결정문을 수집.
사기 사례 RAG의 소스 데이터로 활용.
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cases" / "sec_enforcement"

# SEC EDGAR full-text search API
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ACTIONS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

# SEC 제재 조치 RSS/JSON feed
SEC_LITIGATION_URL = "https://www.sec.gov/cgi-bin/viewer?action=view&cik=&type=LIT&dateb=&owner=include&count=40&search_text=&action=getcompany"

# EDGAR Full-Text Search
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"

HEADERS = {
    "User-Agent": "AIFin Research suanlab@example.com",
    "Accept": "application/json",
}

# 사기 관련 키워드
FRAUD_KEYWORDS = [
    "securities fraud",
    "wire fraud",
    "insider trading",
    "market manipulation",
    "pump and dump",
    "Ponzi scheme",
    "money laundering",
    "account fraud",
    "identity theft",
    "structuring",
]


def search_sec_actions(query: str, start: int = 0, count: int = 10) -> list[dict]:
    """SEC EDGAR full-text search API로 제재 조치 검색."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": "2015-01-01",
        "enddt": "2026-01-01",
        "forms": "LIT",  # Litigation releases
        "from": start,
        "size": count,
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"  Search API error: {e}")
        return []


def fetch_litigation_releases(num_pages: int = 10) -> list[dict]:
    """SEC Litigation Releases 페이지에서 제재 조치 목록 수집."""
    base_url = "https://www.sec.gov/litigation/litreleases"
    releases = []

    for year in range(2024, 2014, -1):
        url = f"{base_url}/litrelarchive/litrelarchive{year}.shtml"
        print(f"  Fetching {year} releases...")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                # 최신 연도는 다른 URL 패턴
                url = f"{base_url}.shtml"
                resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"    Error fetching {year}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/litreleases/" in href and href.endswith(".htm"):
                title = link.get_text(strip=True)
                # 사기 관련 키워드 필터링
                if any(kw.lower() in title.lower() for kw in FRAUD_KEYWORDS):
                    full_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
                    releases.append({
                        "title": title,
                        "url": full_url,
                        "year": year,
                    })

        time.sleep(1)  # rate limit

    print(f"  Found {len(releases)} fraud-related releases")
    return releases


def fetch_release_content(url: str) -> str | None:
    """개별 제재 결정문 전문 수집."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # 본문 추출
        content_div = soup.find("div", {"class": "article-body"}) or soup.find("body")
        if content_div:
            # 불필요한 태그 제거
            for tag in content_div.find_all(["script", "style", "nav"]):
                tag.decompose()
            text = content_div.get_text(separator="\n", strip=True)
            # 정리
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
    return None


def extract_case_metadata(text: str, title: str) -> dict:
    """제재 결정문에서 구조화된 메타데이터 추출."""
    metadata = {"title": title}

    # 금액 추출
    amounts = re.findall(r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion))?", text, re.IGNORECASE)
    if amounts:
        metadata["amounts_mentioned"] = amounts[:5]

    # 사기 유형 분류
    text_lower = text.lower()
    fraud_types = []
    type_mapping = {
        "insider trading": "Insider Trading",
        "pump and dump": "Pump & Dump",
        "ponzi": "Ponzi Scheme",
        "market manipulation": "Market Manipulation",
        "securities fraud": "Securities Fraud",
        "wire fraud": "Wire Fraud",
        "money laundering": "Money Laundering",
        "structuring": "Structuring",
        "identity": "Identity Fraud",
        "account takeover": "Account Takeover",
    }
    for keyword, label in type_mapping.items():
        if keyword in text_lower:
            fraud_types.append(label)
    metadata["fraud_types"] = fraud_types if fraud_types else ["Other"]

    # 관련 법률/규정
    regulations = re.findall(
        r"(?:Section|Rule|Regulation)\s+\d+[a-zA-Z\-()]*(?:\s+of\s+the\s+[\w\s]+Act)?",
        text,
    )
    if regulations:
        metadata["regulations"] = list(set(regulations[:10]))

    return metadata


def save_case(case: dict, idx: int) -> None:
    """사기 사례를 JSON 파일로 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"sec_case_{idx:04d}.json"
    filepath = DATA_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Crawl SEC enforcement actions")
    parser.add_argument("--num_cases", type=int, default=100, help="Target number of cases")
    parser.add_argument("--skip_content", action="store_true", help="Skip fetching full content")
    args = parser.parse_args()

    print("=== SEC Enforcement Actions Crawler ===\n")

    # Step 1: 제재 조치 목록 수집
    print("[1/3] Fetching litigation release listings...")
    releases = fetch_litigation_releases()

    if not releases:
        print("No releases found. Trying keyword search...")
        for keyword in FRAUD_KEYWORDS[:3]:
            hits = search_sec_actions(keyword, count=20)
            for hit in hits:
                source = hit.get("_source", {})
                releases.append({
                    "title": source.get("display_title", ""),
                    "url": source.get("file_url", ""),
                    "year": source.get("file_date", "")[:4],
                })
            time.sleep(1)

    releases = releases[: args.num_cases]
    print(f"\n[2/3] Processing {len(releases)} releases...")

    # Step 2: 전문 수집 + 메타데이터 추출
    cases = []
    for i, release in enumerate(releases):
        print(f"  [{i + 1}/{len(releases)}] {release['title'][:80]}...")

        case = {
            "id": f"SEC-{i + 1:04d}",
            "source": "SEC Litigation Release",
            "title": release["title"],
            "url": release["url"],
            "year": release.get("year"),
        }

        if not args.skip_content and release.get("url"):
            content = fetch_release_content(release["url"])
            if content:
                case["content"] = content
                case["metadata"] = extract_case_metadata(content, release["title"])
                case["content_length"] = len(content)
            time.sleep(0.5)  # rate limit

        cases.append(case)
        save_case(case, i + 1)

    # Step 3: 요약 저장
    summary = {
        "total_cases": len(cases),
        "cases_with_content": sum(1 for c in cases if "content" in c),
        "fraud_type_distribution": {},
    }
    for case in cases:
        for ft in case.get("metadata", {}).get("fraud_types", ["Unknown"]):
            summary["fraud_type_distribution"][ft] = (
                summary["fraud_type_distribution"].get(ft, 0) + 1
            )

    summary_path = DATA_DIR / "crawl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[3/3] Summary:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  With content: {summary['cases_with_content']}")
    print(f"  Fraud types: {summary['fraud_type_distribution']}")
    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
