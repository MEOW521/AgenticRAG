import os
import argparse
import json
import fitz
import re


_RE_PAGE_NUM = re.compile(r"^\d+\s*/\s*\d+$")
_RE_SECTION_ONE_LINE = re.compile(r"^第[一二三四五六七八九十百零两]+节\s+(.+)$")
_RE_SECTION_ONLY = re.compile(r"^第[一二三四五六七八九十百零两]+节\s*$")
_RE_SUBSECTION_HEAD = re.compile(
    r"^(?:[一二三四五六七八九十百]+、|\d+\.\s|\(\d+\)\s*\S)"
)

def extract_text_by_page(pdf_path: str):
    """extract text by page from pdf file

    Args:
        pdf_path (str): path to pdf file

    Returns:
        list[dict]: list of pages with text and page number
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": i + 1, 
                "text": text.strip()
                })
    doc.close()
    return pages


def _strip_toc_dot_leaders(s: str):
    return re.sub(r"\s*\.{3,}.*$", "", s).strip()
def _clean_lines_for_section_scan(text: str):
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _RE_PAGE_NUM.match(line):
            continue
        if "年度报告" in line and "公司" in line and len(line) < 120:
            continue
        out.append(line)
    return out


def detect_section_title(text: str):
    """detect section title from text

    Args:
        text (str): text to detect section title

    Returns:
        str | None: section title or None if not found
    """
    cleaned = _clean_lines_for_section_scan(text)
    max_scan = min(40, len(cleaned))
    for i in range(max_scan):
        line = cleaned[i]
        if _RE_SECTION_ONE_LINE.match(line):
            return _strip_toc_dot_leaders(line)
        if _RE_SECTION_ONLY.match(line):
            if i + 1 >= len(cleaned):
                return _strip_toc_dot_leaders(line)
            nxt = cleaned[i + 1]
            if _RE_SUBSECTION_HEAD.match(nxt):
                return _strip_toc_dot_leaders(line)
            return _strip_toc_dot_leaders(f"{line} {nxt}")
    return None


def chunk_pages(pages: list[dict], chunk_size: int = 500,
                overlap: int = 50, doc_title: str = ""):
    """chunk pages into chunks

    Args:
        pages (list[dict]): list of pages with text and page number
        chunk_size (int, optional): max chars per chunk. Defaults to 500.
        overlap (int, optional): overlap chars. Defaults to 50.
        doc_title (str, optional): document title. Defaults to "".

    Returns:
        list[dict]: list of chunks with text, title, pages and section
    """
    chunks = []
    chunk_id = 0
    current_text = ""
    current_section = ""
    current_pages = []

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]

        # 检测章节
        section = detect_section_title(text)
        if section:
            current_section = section

        # 按段落分割（双换行或单换行+缩进）
        paragraphs = re.split(r'\n(?=\s{2,}|\S)', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前 buffer + 新段落不超限，追加
            if len(current_text) + len(para) <= chunk_size:
                current_text += ("\n" if current_text else "") + para
                if page_num not in current_pages:
                    current_pages.append(page_num)
            else:
                # 保存当前 chunk
                if current_text and len(current_text) >= 50:  # 最小长度
                    chunks.append({
                        "chunk_id": f"fin_{chunk_id:04d}",
                        "text": current_text,
                        "title": f"{doc_title} - {current_section}" if current_section else doc_title,
                        "pages": current_pages[:],
                        "section": current_section,
                    })
                    chunk_id += 1

                # 如果段落本身超长，强制切割
                if len(para) > chunk_size:
                    for start in range(0, len(para), chunk_size - overlap):
                        sub = para[start:start + chunk_size]
                        if len(sub) >= 50:
                            chunks.append({
                                "chunk_id": f"fin_{chunk_id:04d}",
                                "text": sub,
                                "title": f"{doc_title} - {current_section}" if current_section else doc_title,
                                "pages": [page_num],
                                "section": current_section,
                            })
                            chunk_id += 1
                    current_text = ""
                    current_pages = []
                else:
                    # overlap: 保留上一段最后部分
                    current_text = para
                    current_pages = [page_num]

    # 最后一个 chunk
    if current_text and len(current_text) >= 50:
        chunks.append({
            "chunk_id": f"fin_{chunk_id:04d}",
            "text": current_text,
            "title": f"{doc_title} - {current_section}" if current_section else doc_title,
            "pages": current_pages[:],
            "section": current_section,
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="parse pdf to chunks")
    parser.add_argument("--pdf", required=True, help="path to pdf file")
    parser.add_argument("--output", default="data/corpus/corpus_<company_name>.json")
    parser.add_argument("--chunk-size", type=int, default=500, help="max chars per chunk")
    args = parser.parse_args()

    print(f"Parsing PDF: {args.pdf}")
    pages = extract_text_by_page(args.pdf)
    print(f"Extracted {len(pages)} pages with text")

    # 自动检测标题
    for line in pages[0]["text"].split("\n"):
        line = line.strip()
        if "公司" in line and len(line) > 5:
            doc_title = line
            break
    print(f"Document title: {doc_title}")

    # 切块
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, doc_title=doc_title)
    print(f"Generated {len(chunks)} chunks")

    # 统计
    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    sections = set(c["section"] for c in chunks if c["section"])
    print(f"Sections detected: {len(sections)}")
    for s in sorted(sections):
        count = sum(1 for c in chunks if c["section"] == s)
        print(f"  [{count:3d}] {s}")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()