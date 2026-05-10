#!/usr/bin/env python3
"""Collect thesis reference PDFs from Zotero into a single working folder.

Reads:
    paper/references.bib
    paper/manual.bib
    Zotero data dir (auto-discovered)
    ~/Downloads/                 (fallback)
    Obsidian thesis folder       (fallback)

Writes:
    ~/Downloads/thesis_reference_pdfs/{citekey}.pdf
    ~/Downloads/thesis_reference_pdfs/_reference_audit.md

All sources are read-only.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import NamedTuple, Optional

import bibtexparser

WORKSPACE = Path("~/Kod/GitHub/thesis_workspace").expanduser()
PAPER_DIR = WORKSPACE / "paper"
BIB_FILES = [PAPER_DIR / "references.bib", PAPER_DIR / "manual.bib"]
TARGET_DIR = Path("~/Downloads/thesis_reference_pdfs").expanduser()
DOWNLOADS = Path("~/Downloads").expanduser()
OBSIDIAN_THESIS = Path("~/Obsidian/Notes/01_Areas/AREA_Study/THESIS").expanduser()

# Allowed reference file extensions in priority order. The first match wins
# when multiple attachments are available for the same Zotero parent item.
REFERENCE_EXTS = (".pdf", ".epub", ".html", ".htm", ".docx", ".doc", ".txt", ".md")
EXT_PRIORITY = {ext: i for i, ext in enumerate(REFERENCE_EXTS)}


class BibEntry(NamedTuple):
    citekey: str
    surname: str
    year: str
    title: str
    bib_file: str
    cited: bool


class PDFCandidate(NamedTuple):
    path: Path
    surname: str
    year: str
    title: str
    source: str  # "zotero" | "downloads" | "obsidian"


class MatchResult(NamedTuple):
    bib: BibEntry
    pdf: Optional[PDFCandidate]
    confidence: str
    similarity: float


def find_zotero_data_dir() -> Optional[Path]:
    """Locate the Zotero data directory. Honours prefs.js override."""
    candidates: list[Path] = []
    profiles = Path("~/Library/Application Support/Zotero/Profiles").expanduser()
    if profiles.exists():
        for prof in profiles.glob("*/prefs.js"):
            try:
                text = prof.read_text(errors="replace")
            except Exception:
                continue
            m = re.search(r'extensions\.zotero\.dataDir",\s*"([^"]+)"', text)
            if m:
                candidates.append(Path(m.group(1)).expanduser())
    candidates.extend(
        [
            Path("~/Zotero").expanduser(),
            Path("~/Library/Application Support/Zotero").expanduser(),
        ]
    )
    for c in candidates:
        if (c / "zotero.sqlite").exists() and (c / "storage").exists():
            return c
    return None


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def extract_surname_from_citekey(citekey: str) -> str:
    """Better BibTeX style: lowercase surname followed by capitalised words.

    Falls back to the leading [a-z]+ run.
    """
    m = re.match(r"^([a-z]+)", citekey)
    return m.group(1) if m else ""


def first_author_surname(author_field: str) -> str:
    if not author_field:
        return ""
    first = author_field.split(" and ")[0].strip()
    if "," in first:
        return first.split(",", 1)[0].strip()
    parts = [p for p in first.split() if p]
    return parts[-1] if parts else ""


def clean_braces(text: str) -> str:
    return re.sub(r"[{}\\]", "", text or "").strip()


def parse_bib_file(path: Path, cited: set[str]) -> list[BibEntry]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        db = bibtexparser.load(f, parser=parser)
    entries: list[BibEntry] = []
    for e in db.entries:
        citekey = e.get("ID", "")
        if not citekey:
            continue
        author = e.get("author") or e.get("editor") or ""
        year = e.get("year") or ""
        if not year:
            date = e.get("date", "")
            ym = re.search(r"\b(19|20)\d{2}\b", date)
            year = ym.group(0) if ym else ""
        ym = re.search(r"\b(19|20)\d{2}\b", year)
        year = ym.group(0) if ym else ""
        title = clean_braces(e.get("title", ""))
        surname = first_author_surname(clean_braces(author))
        if not surname:
            surname = extract_surname_from_citekey(citekey)
        entries.append(
            BibEntry(
                citekey=citekey,
                surname=surname.lower(),
                year=year,
                title=title.lower(),
                bib_file=path.name,
                cited=citekey in cited,
            )
        )
    return entries


def collect_cited_keys(paper_dir: Path) -> set[str]:
    """Walk every .tex file under paper_dir and collect cite keys.

    Strips line comments (lines starting with %) before parsing so that
    citations inside `% TODO_CITE: ...` markers are not counted.
    Recognises the biblatex/natbib commands the thesis uses.
    """
    cite_cmd = (
        r"\\(?:cite|parencite|textcite|autocite|citep|citet|citeauthor|"
        r"citeyear|citeyearpar|citenum|fullcite|smartcite|footcite)\*?"
        r"(?:\[[^\]]*\]){0,2}\s*\{([^}]+)\}"
    )
    pattern = re.compile(cite_cmd)
    keys: set[str] = set()
    if not paper_dir.exists():
        return keys
    for tex in paper_dir.rglob("*.tex"):
        text = tex.read_text(encoding="utf-8", errors="replace")
        # Strip `%`-comments line-by-line. Escaped `\%` is kept.
        cleaned: list[str] = []
        for line in text.splitlines():
            i = 0
            out_chars: list[str] = []
            while i < len(line):
                if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "%":
                    out_chars.append(line[i : i + 2])
                    i += 2
                elif line[i] == "%":
                    break
                else:
                    out_chars.append(line[i])
                    i += 1
            cleaned.append("".join(out_chars))
        stripped = "\n".join(cleaned)
        for m in pattern.finditer(stripped):
            for key in m.group(1).split(","):
                k = key.strip()
                if k:
                    keys.add(k)
    return keys


def build_zotero_index(zotero_dir: Path) -> list[PDFCandidate]:
    sqlite_path = zotero_dir / "zotero.sqlite"
    storage_dir = zotero_dir / "storage"
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        shutil.copy2(sqlite_path, tmp.name)
        tmp_path = Path(tmp.name)
    try:
        conn = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
        cur = conn.cursor()
        title_field = cur.execute(
            "SELECT fieldID FROM fields WHERE fieldName='title'"
        ).fetchone()
        date_field = cur.execute(
            "SELECT fieldID FROM fields WHERE fieldName='date'"
        ).fetchone()
        if not title_field or not date_field:
            conn.close()
            return []
        title_field_id = title_field[0]
        date_field_id = date_field[0]

        # Map every parent itemID -> (title, date, first author surname)
        meta_sql = f"""
            SELECT i.itemID, i.key,
                   t.value AS title, d.value AS date
            FROM items i
            LEFT JOIN deletedItems del ON del.itemID = i.itemID
            LEFT JOIN itemData td ON td.itemID = i.itemID AND td.fieldID = {title_field_id}
            LEFT JOIN itemDataValues t ON t.valueID = td.valueID
            LEFT JOIN itemData dd ON dd.itemID = i.itemID AND dd.fieldID = {date_field_id}
            LEFT JOIN itemDataValues d ON d.valueID = dd.valueID
            WHERE del.itemID IS NULL
        """
        meta: dict[int, tuple[str, str, str]] = {}
        key_by_id: dict[int, str] = {}
        for item_id, key, title, date in cur.execute(meta_sql):
            key_by_id[item_id] = key
            ym = re.search(r"\b(19|20)\d{2}\b", date or "")
            meta[item_id] = (
                (title or "").strip(),
                ym.group(0) if ym else "",
                "",
            )

        author_sql = """
            SELECT ic.itemID, c.lastName
            FROM itemCreators ic
            JOIN creators c ON c.creatorID = ic.creatorID
            WHERE ic.orderIndex = 0
        """
        for item_id, last_name in cur.execute(author_sql):
            if item_id in meta:
                title, year, _ = meta[item_id]
                meta[item_id] = (title, year, (last_name or "").strip())

        ext_clause = " OR ".join(
            [f"LOWER(ia.path) LIKE '%{ext}'" for ext in REFERENCE_EXTS]
        )
        attach_sql = f"""
            SELECT a.key, ia.parentItemID, ia.path, ia.contentType
            FROM itemAttachments ia
            JOIN items a ON a.itemID = ia.itemID
            LEFT JOIN deletedItems del ON del.itemID = a.itemID
            WHERE del.itemID IS NULL
              AND ia.parentItemID IS NOT NULL
              AND ia.path IS NOT NULL
              AND ({ext_clause})
        """
        # First pass: collect every candidate per parent_id so we can pick the
        # best file format per item (PDF beats HTML beats EPUB, etc.).
        per_parent: dict[int, list[PDFCandidate]] = defaultdict(list)
        for attach_key, parent_id, path, _ctype in cur.execute(attach_sql):
            if parent_id not in meta:
                continue
            title, year, surname = meta[parent_id]
            file_path: Optional[Path] = None
            if path.startswith("storage:"):
                filename = path[len("storage:") :]
                file_path = storage_dir / attach_key / filename
            elif path.startswith("attachments:"):
                continue
            else:
                p = Path(path)
                if p.is_absolute() and p.exists():
                    file_path = p
            if file_path is None or not file_path.exists():
                continue
            per_parent[parent_id].append(
                PDFCandidate(
                    path=file_path,
                    surname=surname.lower(),
                    year=year,
                    title=title.lower(),
                    source="zotero",
                )
            )
        candidates: list[PDFCandidate] = []
        for parent_id, files in per_parent.items():
            files.sort(key=lambda f: EXT_PRIORITY.get(f.path.suffix.lower(), 99))
            candidates.append(files[0])
        conn.close()
        return candidates
    finally:
        tmp_path.unlink(missing_ok=True)


def scan_directory(directory: Path, source: str, exclude: Optional[Path] = None) -> list[PDFCandidate]:
    if not directory.exists():
        return []
    out: list[PDFCandidate] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in REFERENCE_EXTS:
            continue
        if exclude is not None:
            try:
                path.resolve().relative_to(exclude.resolve())
                continue
            except ValueError:
                pass
        name = path.stem
        ym = re.search(r"\b(19|20)\d{2}\b", name)
        year = ym.group(0) if ym else ""
        tokens = re.split(r"[_\s\-,\.]+", name)
        surname = ""
        for t in tokens:
            if (
                t.isalpha()
                and len(t) >= 3
                and t.lower() not in {"the", "and", "for", "from", "with", "vol", "pdf", "html", "epub"}
            ):
                surname = t.lower()
                break
        out.append(
            PDFCandidate(
                path=path,
                surname=surname,
                year=year,
                title=name.lower(),
                source=source,
            )
        )
    return out


def match_entries(entries: list[BibEntry], pdfs: list[PDFCandidate]) -> list[MatchResult]:
    by_sy: dict[tuple[str, str], list[PDFCandidate]] = defaultdict(list)
    by_year: dict[str, list[PDFCandidate]] = defaultdict(list)
    for p in pdfs:
        if p.surname and p.year:
            by_sy[(p.surname, p.year)].append(p)
        if p.year:
            by_year[p.year].append(p)

    source_priority = {"zotero": 0, "downloads": 1, "obsidian": 2}
    results: list[MatchResult] = []
    for e in entries:
        candidates: list[PDFCandidate] = []
        if e.surname and e.year:
            candidates = list(by_sy.get((e.surname, e.year), []))
            # Also include same-year entries with very-similar surname
            for p in by_year.get(e.year, []):
                if p in candidates or not p.surname:
                    continue
                if SequenceMatcher(None, e.surname, p.surname).ratio() > 0.85:
                    candidates.append(p)
        if not candidates and e.year:
            # Year-only fallback: keep only those with title similarity above 0.5
            for p in by_year.get(e.year, []):
                if title_similarity(e.title, p.title) > 0.5:
                    candidates.append(p)
        if not candidates:
            # Last ditch: title-only match across ALL years (limited)
            for p in pdfs:
                if title_similarity(e.title, p.title) > 0.7:
                    candidates.append(p)

        if not candidates:
            results.append(MatchResult(e, None, "missing", 0.0))
            continue

        scored = [
            (
                title_similarity(e.title, p.title),
                source_priority.get(p.source, 9),
                EXT_PRIORITY.get(p.path.suffix.lower(), 99),
                p,
            )
            for p in candidates
        ]
        # Sort: highest similarity first, then preferred source, then preferred
        # file format (PDF > EPUB > HTML > DOCX > ...).
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        sim, _, _, best = scored[0]

        if sim >= 0.7:
            confidence = "exact" if best.source == "zotero" else "fuzzy_title"
        elif sim >= 0.5:
            confidence = "fuzzy_title"
        elif e.surname and e.year:
            confidence = "surname_year_only"
        else:
            confidence = "low"
        results.append(MatchResult(e, best, confidence, sim))
    return results


def safe_filename(citekey: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", citekey)


def copy_matches(matches: list[MatchResult], target: Path) -> dict[str, int]:
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for m in matches:
        if m.pdf is None:
            continue
        # Remove stale prior-run copies for this citekey before writing the
        # current one (e.g. a `.pdf` from an earlier run when the current
        # match is `.html`).
        stem = safe_filename(m.bib.citekey)
        for stale in target.glob(f"{stem}.*"):
            if stale.suffix.lower() in REFERENCE_EXTS:
                stale.unlink()
        ext = m.pdf.path.suffix.lower()
        if ext not in REFERENCE_EXTS:
            ext = ".pdf"
        dest = target / f"{stem}{ext}"
        try:
            shutil.copy2(m.pdf.path, dest)
            counts[m.pdf.source] += 1
            counts[f"ext_{ext}"] += 1
        except Exception as ex:
            counts["copy_error"] += 1
            print(f"WARN: failed to copy {m.bib.citekey}: {ex}", file=sys.stderr)
    return dict(counts)


def best_zotero_candidate_for(
    entry: BibEntry, zotero_pdfs: list[PDFCandidate]
) -> Optional[tuple[PDFCandidate, float]]:
    """Find the closest Zotero PDF for a bib entry, regardless of threshold."""
    if not zotero_pdfs:
        return None
    scored = [(title_similarity(entry.title, p.title), p) for p in zotero_pdfs]
    scored.sort(key=lambda x: -x[0])
    sim, best = scored[0]
    if sim < 0.3:
        return None
    return best, sim


def write_audit(
    matches: list[MatchResult],
    target: Path,
    source_counts: dict[str, int],
    zotero_dir: Optional[Path],
    zotero_pdfs: list[PDFCandidate],
    orphan_cites: list[str],
) -> None:
    found = [m for m in matches if m.pdf is not None]
    missing = [m for m in matches if m.pdf is None]

    cited_total = sum(1 for m in matches if m.bib.cited)
    cited_found = sum(1 for m in found if m.bib.cited)
    cited_missing = sum(1 for m in missing if m.bib.cited)
    uncited_found = sum(1 for m in found if not m.bib.cited)
    uncited_missing = sum(1 for m in missing if not m.bib.cited)

    def short(t: str, n: int = 60) -> str:
        return t if len(t) <= n else t[: n - 3] + "..."

    home = str(Path.home())

    def rel_path(p: Path) -> str:
        return str(p).replace(home, "~")

    def cite_marker(cited: bool) -> str:
        return "✓" if cited else "—"

    lines: list[str] = []
    lines.append("# Reference PDF collection audit")
    lines.append("")
    lines.append(f"- Total bib entries: **{len(matches)}** "
                 f"(cited in thesis: **{cited_total}**, uncited: {len(matches) - cited_total})")
    lines.append(f"- Found: **{len(found)}** "
                 f"(cited: **{cited_found}**, uncited: {uncited_found})")
    lines.append(f"- Missing: **{len(missing)}** "
                 f"(cited: **{cited_missing}** ← priority, uncited: {uncited_missing})")
    lines.append(f"- Source breakdown: `{dict(source_counts)}`")
    lines.append(
        f"- Zotero data dir: `{zotero_dir}`"
        if zotero_dir
        else "- Zotero data dir: NOT FOUND"
    )
    lines.append("")

    if orphan_cites:
        lines.append("## Orphan cite keys")
        lines.append("")
        lines.append(
            "Cite keys used in the LaTeX but absent from `references.bib` and "
            "`manual.bib`. These will fail biblatex resolution at compile time."
        )
        lines.append("")
        for k in orphan_cites:
            lines.append(f"- `{k}`")
        lines.append("")

    lines.append("## MISSING — cited in thesis (priority to fetch)")
    lines.append("")
    cited_missing_rows = [m for m in missing if m.bib.cited]
    if cited_missing_rows:
        lines.append("| Citekey | Year | Title | Bib | Closest Zotero candidate (sim) |")
        lines.append("|---|---|---|---|---|")
        for m in sorted(cited_missing_rows, key=lambda x: x.bib.citekey):
            cand = best_zotero_candidate_for(m.bib, zotero_pdfs)
            cand_str = f"{short(cand[0].title, 50)} ({cand[1]:.2f})" if cand else "—"
            lines.append(
                f"| `{m.bib.citekey}` | {m.bib.year or '—'} | "
                f"{short(m.bib.title)} | {m.bib.bib_file} | {cand_str} |"
            )
    else:
        lines.append("_(none — every cited reference has a file)_")
    lines.append("")

    lines.append("## MISSING — not cited (no action needed)")
    lines.append("")
    uncited_missing_rows = [m for m in missing if not m.bib.cited]
    if uncited_missing_rows:
        lines.append("| Citekey | Year | Title | Bib |")
        lines.append("|---|---|---|---|")
        for m in sorted(uncited_missing_rows, key=lambda x: x.bib.citekey):
            lines.append(
                f"| `{m.bib.citekey}` | {m.bib.year or '—'} | "
                f"{short(m.bib.title)} | {m.bib.bib_file} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("## FOUND — cited in thesis")
    lines.append("")
    lines.append("Sorted: low-confidence matches first (verify these).")
    lines.append("")
    cited_found_rows = [m for m in found if m.bib.cited]
    lines.append("| Citekey | Year | Title | Source | Conf. | Sim. | Bib | Original path |")
    lines.append("|---|---|---|---|---|---|---|---|")
    conf_order = {"surname_year_only": 0, "low": 1, "fuzzy_title": 2, "exact": 3}
    for m in sorted(
        cited_found_rows,
        key=lambda x: (conf_order.get(x.confidence, 9), x.similarity, x.bib.citekey),
    ):
        assert m.pdf is not None
        lines.append(
            f"| `{m.bib.citekey}` | {m.bib.year or '—'} | "
            f"{short(m.bib.title)} | {m.pdf.source} | {m.confidence} | "
            f"{m.similarity:.2f} | {m.bib.bib_file} | `{rel_path(m.pdf.path)}` |"
        )
    lines.append("")

    lines.append("## FOUND — not cited (in bib but unused in thesis)")
    lines.append("")
    uncited_found_rows = [m for m in found if not m.bib.cited]
    if uncited_found_rows:
        lines.append("| Citekey | Year | Title | Source | Bib |")
        lines.append("|---|---|---|---|---|")
        for m in sorted(uncited_found_rows, key=lambda x: x.bib.citekey):
            assert m.pdf is not None
            lines.append(
                f"| `{m.bib.citekey}` | {m.bib.year or '—'} | "
                f"{short(m.bib.title)} | {m.pdf.source} | {m.bib.bib_file} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    audit = target / "_reference_audit.md"
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("=== Thesis reference PDF collector ===")

    zotero_dir = find_zotero_data_dir()
    if zotero_dir:
        print(f"Zotero data dir: {zotero_dir}")
    else:
        print("WARN: Zotero data directory not found; relying on filesystem fallbacks")

    cited = collect_cited_keys(PAPER_DIR)
    print(f"Collected {len(cited)} unique cite keys from .tex sources")

    entries: list[BibEntry] = []
    for bf in BIB_FILES:
        parsed = parse_bib_file(bf, cited)
        print(f"Parsed {len(parsed)} entries from {bf.name}")
        entries.extend(parsed)
    if not entries:
        print("ERROR: no bib entries parsed", file=sys.stderr)
        return 1

    bib_keys = {e.citekey for e in entries}
    orphan_cites = sorted(cited - bib_keys)
    if orphan_cites:
        print(f"WARN: {len(orphan_cites)} cite keys used in .tex but absent from .bib")
    cited_count = sum(1 for e in entries if e.cited)
    print(f"Of {len(entries)} bib entries, {cited_count} are cited in the thesis")

    pdfs: list[PDFCandidate] = []
    if zotero_dir:
        z = build_zotero_index(zotero_dir)
        print(f"Indexed {len(z)} PDFs in Zotero storage")
        pdfs.extend(z)

    d = scan_directory(DOWNLOADS, "downloads", exclude=TARGET_DIR)
    print(f"Indexed {len(d)} PDFs in ~/Downloads/ (excluding target dir)")
    pdfs.extend(d)

    o = scan_directory(OBSIDIAN_THESIS, "obsidian")
    print(f"Indexed {len(o)} PDFs in Obsidian thesis folder")
    pdfs.extend(o)

    matches = match_entries(entries, pdfs)
    source_counts = copy_matches(matches, TARGET_DIR)
    zotero_pdfs = [p for p in pdfs if p.source == "zotero"]
    write_audit(
        matches, TARGET_DIR, source_counts, zotero_dir, zotero_pdfs, orphan_cites
    )

    found = sum(1 for m in matches if m.pdf is not None)
    print(f"Audit: {TARGET_DIR / '_reference_audit.md'}")
    print(f"Found: {found}/{len(matches)} | Missing: {len(matches) - found}/{len(matches)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
