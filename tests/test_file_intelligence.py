"""Reading files without reading all of them.

The expensive mistake this whole layer exists to avoid is putting a folder of
contracts into the prompt to answer a question about one number in one of them.
So most of what is checked here is what does NOT come back: files that were not
relevant, sections that did not answer the question, and the rest of a file when
part of it was enough.

The rest is honesty. Every piece of text that comes back carries the label of the
place it came from, because a quotation nobody can locate is worth less than no
quotation; and a file that cannot be read comes back as a refusal with a reason,
never as an empty extraction that reads like an empty document.

The fixtures are built here rather than committed: a real PDF and a real .docx,
written by hand out of the standard library, so the tests exercise the actual
parsers rather than a stub.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from core import documents

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- Fixtures that are really files ---------------------------------------------

def build_pdf(pages) -> bytes:
    """A minimal, valid PDF whose text is genuinely extractable."""
    objects = []
    page_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects.append((2, b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(pages)))
    for i, lines in enumerate(pages):
        pid, cid = page_ids[i], page_ids[i] + 1
        objects.append((pid, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                             b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                             % (font_id, cid)))
        body = [b"BT", b"/F1 12 Tf", b"72 720 Td", b"14 TL"]
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body.append(b"(" + escaped.encode("latin-1", "replace") + b") Tj T*")
        body.append(b"ET")
        stream = b"\n".join(body)
        objects.append((cid, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"))
    objects.append((font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number, payload in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + payload + b"\nendobj\n"
    start, count = len(out), max(offsets) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % count
    for number in range(1, count):
        out += b"%010d 00000 n \n" % offsets.get(number, 0)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (count, start)
    return bytes(out)


_DOCX_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _docx_paragraph(style, text) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;")
    return f"<w:p>{style_xml}<w:r><w:t>{safe}</w:t></w:r></w:p>"


def build_docx(path: Path, blocks) -> None:
    """A real .docx: a zip of XML, which is all it ever was.

    A block is ("Heading1", "text"), ("", "text"), or ("table", [[cells], ...]).
    """
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = []
    for style, text in blocks:
        if style == "table":
            rows = "".join(
                "<w:tr>" + "".join(f"<w:tc>{_docx_paragraph('', cell)}</w:tc>"
                                   for cell in row) + "</w:tr>"
                for row in text)
            body.append(f"<w:tbl>{rows}</w:tbl>")
            continue
        body.append(_docx_paragraph(style, text))
    document = (f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{namespace}">'
                f'<w:body>{"".join(body)}</w:body></w:document>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _DOCX_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        archive.writestr("word/document.xml", document)


@pytest.fixture
def folder(tmp_path):
    """An offers folder of the shape the request describes."""
    (tmp_path / "Offer_Company_A.pdf").write_bytes(build_pdf([
        ["Offer from Company A", "Scope of works and introduction."],
        ["Delivery schedule: eight weeks from order."],
        ["Pricing", "Total price: 42,500 EUR", "Payment terms: 30 days net"],
    ]))
    build_docx(tmp_path / "Contract_B.docx", [
        ("Heading1", "Scope"), ("", "Company B will deliver the same works."),
        ("Heading1", "Pricing table"), ("", "Total price: 39,900 EUR"),
        ("", "Payment terms: 60 days net"),
        ("Heading1", "Warranty"), ("", "Two years on parts and labour."),
    ])
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "Summary"
    sheet.append(["Vendor", "Price", "Weeks"])
    sheet.append(["A", 42500, 8])
    sheet.append(["B", 39900, 12])
    book.create_sheet("Blank").append(["nothing of interest"])
    book.save(tmp_path / "comparison.xlsx")

    (tmp_path / "holiday_photos_readme.txt").write_text(
        "These are pictures from the summer trip. Nothing to do with work.\n")
    (tmp_path / "settings.json").write_text(json.dumps(
        {"pricing": {"total": 42500, "currency": "EUR"}, "colours": ["red", "blue"]}))
    (tmp_path / "notes.md").write_text(
        "# Background\nSome history.\n\n# Payment\nTotal price is the deciding factor.\n")
    return tmp_path


# --- One file ---------------------------------------------------------------------

def test_a_pdf_is_read_by_the_page_that_answers_the_question(folder):
    result = documents.extract(folder / "Offer_Company_A.pdf", "total price payment terms")
    assert result["ok"] is True
    assert [s["label"] for s in result["sections"]] == ["Page 3"]
    assert "42,500" in result["sections"][0]["text"]
    # And the rest of the document did NOT come back.
    assert result["sections_total"] == 3
    assert result["more_available"] is True
    assert "Scope of works" not in json.dumps(result)


def test_a_word_document_is_read_by_its_own_headings(folder):
    result = documents.extract(folder / "Contract_B.docx", "payment terms")
    assert [s["label"] for s in result["sections"]] == ["Pricing table"]
    assert "39,900" in result["sections"][0]["text"]
    assert "Warranty" not in json.dumps(result["sections"])


def test_a_spreadsheet_is_read_by_sheet(folder):
    result = documents.extract(folder / "comparison.xlsx", "vendor price")
    assert [s["label"] for s in result["sections"]] == ["Sheet 'Summary'"]
    assert "39900" in result["sections"][0]["text"]


@pytest.mark.parametrize("name,query,expected", [
    ("notes.md", "payment price", "# Payment"),
    ("settings.json", "pricing total", "key 'pricing'"),
])
def test_every_kind_carries_a_label_from_the_file_itself(folder, name, query, expected):
    result = documents.extract(folder / name, query)
    assert result["sections"][0]["label"] == expected


# --- Large files ------------------------------------------------------------------

def test_a_large_file_comes_back_as_the_rows_that_matched(tmp_path):
    """Ten thousand rows, and the answer is one of them."""
    rows = ["vendor,price,weeks"] + [f"v{i},{1000 + i},{i % 20}" for i in range(10_000)]
    big = tmp_path / "rates.csv"
    big.write_text("\n".join(rows))

    result = documents.extract(big, "v9481")
    assert result["ok"] is True
    assert result["characters"] <= documents.DEFAULT_EXTRACT_CHARS
    assert result["characters"] < big.stat().st_size / 10
    assert "v9481" in result["sections"][0]["text"]
    assert result["sections"][0]["label"].startswith("Rows ")
    assert result["more_available"] is True


def test_asking_for_a_named_section_reads_only_that_one(folder):
    result = documents.extract(folder / "Offer_Company_A.pdf", section="Page 2")
    assert [s["label"] for s in result["sections"]] == ["Page 2"]
    assert "eight weeks" in result["sections"][0]["text"]


def test_a_section_that_does_not_exist_says_which_ones_do(folder):
    result = documents.extract(folder / "Offer_Company_A.pdf", section="Pricing")
    assert result["ok"] is False
    assert "Page 1" in result["available"]


def test_a_page_past_the_end_says_how_many_there_are(folder):
    """And does not say "this looks like a scan", which is a different problem."""
    result = documents.extract(folder / "Offer_Company_A.pdf", section="Page 99")
    assert result["ok"] is False
    assert "has 3 page(s)" in result["error"]
    assert "scan" not in result["error"].lower()


def test_the_extraction_budget_is_honoured(tmp_path):
    long_text = tmp_path / "long.txt"
    long_text.write_text("\n".join(f"line {i} about pricing" for i in range(20_000)))
    result = documents.extract(long_text, "pricing", max_chars=1_200)
    assert result["characters"] <= 1_200


# --- Which files ------------------------------------------------------------------

def test_the_irrelevant_file_is_not_chosen(folder):
    files = documents.candidate_files(folder)
    ranked = documents.find_relevant("compare the offers and contracts on price", files)
    matched = [f["name"] for f in ranked if f["score"] > 0]
    assert "holiday_photos_readme.txt" not in matched
    assert any(name.startswith("Offer_") or name.startswith("Contract_") for name in matched)


def test_choosing_files_opens_none_of_them(folder, monkeypatch):
    """Ranking by name costs nothing; opening a folder to rank it would not."""
    def refuse(*a, **k):
        raise AssertionError("find_relevant opened a file to rank it")

    monkeypatch.setattr(documents, "sections", refuse)
    ranked = documents.find_relevant("offers", documents.candidate_files(folder))
    assert ranked


def test_looking_inside_is_opt_in_and_finds_what_names_do_not(folder):
    """'warranty' appears in no filename. It is inside one of them."""
    files = documents.candidate_files(folder)
    by_name = documents.find_relevant("warranty on parts and labour", files)
    assert all(f["score"] == 0 for f in by_name)

    deeper = documents.find_relevant("warranty on parts and labour", files, look_inside=6)
    best = [f for f in deeper if f["score"] > 0]
    assert best and best[0]["name"] == "Contract_B.docx"
    assert "inside" in best[0]["why"]


# --- Things that cannot be read ----------------------------------------------------

def test_an_unsupported_kind_is_refused_rather_than_guessed(tmp_path):
    odd = tmp_path / "archive.zip"
    odd.write_bytes(b"PK\x03\x04nonsense")
    result = documents.describe(odd)
    assert result["ok"] is False
    assert "no reader" in result["error"]


def test_a_corrupt_file_reports_the_problem_and_returns_no_text(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    result = documents.extract(broken, "price")
    assert result["ok"] is False
    assert result.get("sections") is None
    assert "broken.pdf" in result["error"]


def test_a_missing_file_is_a_missing_file(tmp_path):
    result = documents.extract(tmp_path / "nope.pdf", "price")
    assert result["ok"] is False and "no file" in result["error"]


def test_an_empty_extraction_is_an_error_not_an_empty_document(tmp_path):
    """A scanned PDF has pages and no text. Reporting that as an empty document is
    how a confident summary of nothing gets written."""
    blank = tmp_path / "scan.pdf"
    blank.write_bytes(build_pdf([[], []]))
    result = documents.extract(blank, "anything")
    assert result["ok"] is False
    assert "no extractable text" in result["error"]
    # It says what that means, and that Leti cannot get round it.
    assert "scanned" in result["error"] and "no OCR" in result["error"]
    assert result.get("sections") is None


# --- The tools ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_find_documents_reports_names_and_never_contents(folder):
    from tools.documents import FindDocumentsTool

    result = await FindDocumentsTool().run("compare the offers", folder=str(folder))
    assert result.success
    blob = json.dumps(result.output)
    assert "Offer_Company_A.pdf" in blob
    for secret in ("42,500", "Payment terms", "Scope of works"):
        assert secret not in blob, "a file's contents reached the caller of find_documents"


@pytest.mark.asyncio
async def test_inspect_document_shows_structure_not_text(folder):
    from tools.documents import InspectDocumentTool

    result = await InspectDocumentTool().run(str(folder / "Offer_Company_A.pdf"))
    assert result.success
    assert result.output["pages"] == 3
    assert [s["label"] for s in result.output["sections"]] == ["Page 1", "Page 2", "Page 3"]
    # Previews, not pages.
    assert all(len(s["preview"]) <= documents.OUTLINE_PREVIEW_CHARS
               for s in result.output["sections"])


@pytest.mark.asyncio
async def test_compare_documents_answers_once_per_file_with_its_own_labels(folder):
    from tools.documents import CompareDocumentsTool

    result = await CompareDocumentsTool().run(
        paths=[str(folder / "Offer_Company_A.pdf"), str(folder / "Contract_B.docx")],
        question="total price and payment terms")
    assert result.success
    by_file = {c["file"]: c for c in result.output["compared"]}
    assert set(by_file) == {"Offer_Company_A.pdf", "Contract_B.docx"}
    assert by_file["Offer_Company_A.pdf"]["found"][0]["label"] == "Page 3"
    assert by_file["Contract_B.docx"]["found"][0]["label"] == "Pricing table"
    assert "42,500" in json.dumps(by_file["Offer_Company_A.pdf"])
    assert "39,900" in json.dumps(by_file["Contract_B.docx"])


@pytest.mark.asyncio
async def test_one_unreadable_file_does_not_stop_the_comparison(folder, tmp_path):
    from tools.documents import CompareDocumentsTool

    broken = tmp_path / "torn.pdf"
    broken.write_bytes(b"not a pdf")
    result = await CompareDocumentsTool().run(
        paths=[str(folder / "Offer_Company_A.pdf"), str(broken)],
        question="total price")
    assert result.success
    assert [c["file"] for c in result.output["compared"]] == ["Offer_Company_A.pdf"]
    assert result.output["unreadable"][0]["file"] == "torn.pdf"
    assert "how_to_cite" in result.output


@pytest.mark.asyncio
async def test_every_answer_says_how_to_attribute_it(folder):
    from tools.documents import ReadDocumentTool

    result = await ReadDocumentTool().run(str(folder / "Offer_Company_A.pdf"),
                                          question="payment terms")
    assert result.success
    assert "Offer_Company_A.pdf" in result.output["how_to_cite"]
    assert "Do not cite a label that is not listed" in result.output["how_to_cite"]
    assert result.output["source"] == "Offer_Company_A.pdf"


# --- Project scope --------------------------------------------------------------------

@pytest.fixture
def project(tmp_path, monkeypatch):
    from tools import projects

    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(projects, "projects_root", lambda: root)
    monkeypatch.setattr(projects, "_state_path", lambda: tmp_path / "active.json")
    projects.project_dir("Parot").mkdir(parents=True, exist_ok=True)
    projects.save_manifest("Parot", {"name": "Parot", "description": "", "instructions": ""})
    (projects.project_dir("Parot") / "invoice_2024_11.pdf").write_bytes(
        build_pdf([["Invoice 2024-11", "Amount due: 1,200 EUR"]]))
    (tmp_path / "elsewhere.pdf").write_bytes(build_pdf([["Invoice from somewhere else"]]))
    return projects


@pytest.mark.asyncio
async def test_an_open_project_is_where_files_are_looked_for(project):
    from tools.documents import FindDocumentsTool

    project.set_active_project("Parot")
    result = await FindDocumentsTool().run("find the invoice")
    assert result.success
    assert result.output["project"] == "Parot"
    names = [f["name"] for f in result.output["files"]]
    assert names == ["invoice_2024_11.pdf"], "it searched beyond the open project"


@pytest.mark.asyncio
async def test_with_no_project_and_no_folder_it_asks_rather_than_searching_the_disk(project):
    from tools.documents import FindDocumentsTool

    project.set_active_project(None)
    result = await FindDocumentsTool().run("find the invoice")
    assert result.success is False
    assert "no project is open" in result.error


@pytest.mark.asyncio
async def test_a_named_folder_overrides_the_open_project(project, tmp_path):
    from tools.documents import FindDocumentsTool

    project.set_active_project("Parot")
    result = await FindDocumentsTool().run("find the invoice", folder=str(tmp_path))
    assert result.success
    assert result.output["project"] is None
    assert "elsewhere.pdf" in [f["name"] for f in result.output["files"]]


# --- PDFs that are not the easy case -------------------------------------------------

def test_a_long_pdf_is_read_in_a_window_and_says_which_pages(tmp_path):
    """Extracting text from four hundred pages to answer one question is minutes of
    work. A window is read, and the answer says so - otherwise "not in this
    document" would really mean "not in the pages that were opened"."""
    pages = [[f"Page {i} of ordinary filler"] for i in range(1, 301)]
    pages[249] = ["The penalty clause is 5% per week."]
    long_pdf = tmp_path / "long.pdf"
    long_pdf.write_bytes(build_pdf(pages))

    result = documents.extract(long_pdf, "penalty clause")
    assert result["ok"] is True
    assert result["pages"] == 300
    assert result["pages_scanned"] == documents.MAX_PDF_PAGES_SCANNED
    assert "Pages 1-120 of 300 were read" in result["note"]
    # The answer is on page 250, which was not opened - and nothing pretends it was.
    assert "penalty" not in json.dumps(result["sections"])


def test_a_named_page_opens_that_page_rather_than_reading_up_to_it(tmp_path):
    pages = [[f"Page {i}"] for i in range(1, 301)]
    pages[249] = ["The penalty clause is 5% per week."]
    long_pdf = tmp_path / "long.pdf"
    long_pdf.write_bytes(build_pdf(pages))

    result = documents.extract(long_pdf, section="Page 250")
    assert [s["label"] for s in result["sections"]] == ["Page 250"]
    assert "penalty clause" in result["sections"][0]["text"]
    assert result["first_page_scanned"] == 250


@pytest.mark.parametrize("asked,expected", [
    ("Page 2-3", ["Page 2", "Page 3"]),
    ("Page 1 to 2", ["Page 1", "Page 2"]),
])
def test_a_range_of_pages_can_be_asked_for(folder, asked, expected):
    result = documents.extract(folder / "Offer_Company_A.pdf", section=asked)
    assert [s["label"] for s in result["sections"]] == expected


def test_a_scanned_pdf_says_it_is_a_scan_and_offers_nothing_else(tmp_path):
    """The one place a confident summary of nothing could come from."""
    scan = tmp_path / "scanned.pdf"
    scan.write_bytes(build_pdf([[], [], []]))
    result = documents.extract(scan, "the total")
    assert result["ok"] is False
    assert "no extractable text" in result["error"]
    assert "scanned" in result["error"] and "no OCR" in result["error"]
    assert result.get("sections") is None


def test_a_pdf_that_is_partly_text_reports_the_pages_that_are_not(tmp_path):
    mixed = tmp_path / "mixed.pdf"
    mixed.write_bytes(build_pdf([["The readable page."], [], []]))
    result = documents.extract(mixed, "readable")
    assert result["ok"] is True
    assert result["pages_without_text"] == 2
    assert "no extractable text" in result["note"] and "no OCR" in result["note"]


def test_a_pdf_with_a_damaged_structure_is_still_read_where_it_can_be(tmp_path):
    """strict=False: a broken cross-reference table is not a reason to refuse a
    document whose pages are perfectly readable."""
    good = build_pdf([["Total price: 42,500 EUR"]])
    damaged = tmp_path / "damaged.pdf"
    # Corrupt the xref offsets, which is the commonest real-world damage.
    damaged.write_bytes(good.replace(b"0000000009", b"0000000999"))
    result = documents.extract(damaged, "total price")
    assert result["ok"] is True
    assert "42,500" in result["sections"][0]["text"]


def test_a_pdf_that_is_not_a_pdf_is_refused_with_a_reason(tmp_path):
    broken = tmp_path / "torn.pdf"
    broken.write_bytes(b"%PDF-1.4 and then nothing that parses")
    result = documents.extract(broken, "anything")
    assert result["ok"] is False
    assert "torn.pdf" in result["error"]
    assert result.get("sections") is None


# --- DOCX tables -----------------------------------------------------------------------

@pytest.fixture
def tabled(tmp_path):
    build_docx(tmp_path / "quote.docx", [
        ("Heading1", "Scope"), ("", "Company B will do the works."),
        ("Heading1", "Pricing table"),
        ("table", [["Item", "Qty", "Unit price", "Total"],
                   ["Design", "1", "4,000", "4,000"],
                   ["Build", "1", "32,000", "32,000"],
                   ["Warranty", "2", "1,950", "3,900"]]),
        ("", "All figures exclude VAT."),
        ("Heading1", "Warranty"), ("", "Two years."),
    ])
    return tmp_path / "quote.docx"


def test_a_table_comes_back_as_rows_and_cells(tabled):
    parts = {s["label"]: s["text"] for s in documents.sections(tabled)}
    table = next(text for label, text in parts.items() if label.startswith("Table 1"))
    rows = [line.split("\t") for line in table.splitlines()]
    assert rows[0] == ["Item", "Qty", "Unit price", "Total"]
    assert ["Build", "1", "32,000", "32,000"] in rows
    assert len(rows) == 4


def test_a_table_is_labelled_by_where_it_stands_in_the_document(tabled):
    labels = [s["label"] for s in documents.sections(tabled)]
    assert any(label.startswith("Table 1 under 'Pricing table'") for label in labels)
    # And it is in document order: after Scope, before Warranty.
    assert labels.index("Scope") < next(
        i for i, l in enumerate(labels) if l.startswith("Table 1"))
    assert next(i for i, l in enumerate(labels) if l.startswith("Table 1")) < labels.index(
        "Warranty")


def test_cell_text_is_not_also_reported_as_prose(tabled):
    """It used to be: the walk reached the paragraphs inside a table, so the cells
    came back as loose sentences AND the table was only a count."""
    prose = [s for s in documents.sections(tabled) if not s["label"].startswith("Table")]
    assert not any("32,000" in s["text"] for s in prose)


def test_a_question_about_a_table_finds_the_table(tabled):
    result = documents.extract(tabled, "unit price for the build")
    assert result["sections"][0]["label"].startswith("Table 1")
    assert "32,000" in result["sections"][0]["text"]


def test_the_table_count_still_counts_tables(tabled):
    info = documents.describe(tabled)
    assert info["tables"] == 1
    assert info["paragraphs"] == 6      # the cells are not paragraphs any more


def test_a_long_table_is_chunked_with_its_header_repeated(tmp_path):
    rows = [["Row", "Value"]] + [[f"r{i}", str(i)] for i in range(1, 121)]
    build_docx(tmp_path / "big.docx", [("Heading1", "Data"), ("table", rows)])
    parts = [s for s in documents.sections(tmp_path / "big.docx")
             if s["label"].startswith("Table")]
    # 121 rows at 40 to a chunk: 1-40, 41-80, 81-120, and the odd last row.
    assert len(parts) == 4, [p["label"] for p in parts]
    assert "rows 41-80" in parts[1]["label"]
    assert parts[1]["text"].splitlines()[0] == "Row\tValue", "a cited chunk lost its header"


# --- Context protection ------------------------------------------------------------------

def test_an_outline_is_structure_and_stays_small(tmp_path):
    """Forty previews at 140 characters is 5KB of "structure" - which is contents."""
    pages = [[f"Page {i}: " + "long filler text " * 40] for i in range(1, 200)]
    long_pdf = tmp_path / "long.pdf"
    long_pdf.write_bytes(build_pdf(pages))

    info = documents.outline(long_pdf)
    assert len(info["sections"]) <= documents.MAX_OUTLINE_SECTIONS
    previews = sum(len(s["preview"]) for s in info["sections"])
    assert previews <= documents.MAX_OUTLINE_CHARS
    assert info["sections_not_shown"] > 0
    assert len(json.dumps(info)) < 8_000


@pytest.mark.asyncio
async def test_a_comparison_of_many_files_has_a_total_budget_not_just_a_per_file_one(tmp_path):
    from tools.documents import CompareDocumentsTool

    paths = []
    for n in range(8):
        path = tmp_path / f"offer_{n}.pdf"
        path.write_bytes(build_pdf([["Pricing " + "detail " * 400,
                                     f"Total price: {10_000 + n} EUR"]]))
        paths.append(str(path))

    result = await CompareDocumentsTool().run(paths=paths, question="total price",
                                              max_characters_each=4_000)
    assert result.success
    assert result.output["characters"] <= documents.MAX_COMPARE_CHARS
    # And a file that did not fit is named rather than silently dropped.
    skipped = [u for u in result.output["unreadable"] if "size limit" in u["problem"]]
    assert len(result.output["compared"]) + len(skipped) == 8


# --- Images: three different things, one of them absent ------------------------------------

@pytest.fixture
def picture(tmp_path):
    from PIL import Image

    path = tmp_path / "receipt.png"
    Image.new("RGB", (640, 480), (255, 255, 255)).save(path)
    return path


def test_an_image_reports_its_metadata(picture):
    info = documents.describe(picture)
    assert info["ok"] is True
    assert info["kind"] == "image"
    assert (info["width"], info["height"]) == (640, 480)
    assert info["format"] == "PNG"


def test_an_image_says_plainly_that_its_text_cannot_be_extracted(picture):
    info = documents.describe(picture)
    assert info["text_extraction"] == "unavailable"
    assert "no OCR" in info["text_extraction_note"]
    assert info["image_understanding"] == "look_at_image"


def test_reading_an_image_as_a_document_returns_no_text_and_says_why(picture):
    result = documents.extract(picture, "what does the total say")
    assert result["ok"] is True
    body = result["sections"][0]["text"]
    assert "no extractable text" in body and "no OCR" in body
    assert "look_at_image" in body


@pytest.mark.asyncio
async def test_read_file_on_an_image_refuses_rather_than_decoding_pixels(picture):
    from tools.file_manager import ReadFileTool

    result = await ReadFileTool().run(str(picture))
    assert result.success is False
    assert "no OCR" in result.error and "look_at_image" in result.error


@pytest.mark.asyncio
async def test_look_at_image_labels_its_answer_as_a_description_not_as_text(picture):
    from tools.documents import LookAtImageTool

    class FakeVision:
        async def analyze_image(self, question, encoded):
            return "A receipt on a white background."

    result = await LookAtImageTool(FakeVision()).run(str(picture), "what text does it show")
    assert result.success
    assert result.output["kind"] == "image understanding"
    assert "no OCR" in result.output["text_extraction"]
    assert "not text extracted from the file" in result.output["how_to_cite"]
    assert "never as the file's contents" in result.output["how_to_cite"]


@pytest.mark.asyncio
async def test_look_at_image_refuses_a_file_that_is_not_an_image(folder):
    from tools.documents import LookAtImageTool

    result = await LookAtImageTool(object()).run(
        str(folder / "Offer_Company_A.pdf"), "what does it say")
    assert result.success is False
    assert "read_document" in result.error


# --- read_file and read_document are one capability, not two competing ones ----------------

@pytest.mark.asyncio
async def test_read_file_on_a_pdf_returns_its_text_not_its_bytes(folder):
    """It used to decode the file with errors='replace' and report success, handing
    the model a page of mojibake that reads like an unknowable file."""
    from tools.file_manager import ReadFileTool

    result = await ReadFileTool().run(str(folder / "Offer_Company_A.pdf"))
    assert result.success
    assert "%PDF" not in json.dumps(result.output)
    assert [s["label"] for s in result.output["sections"]][0].startswith("Page ")


@pytest.mark.asyncio
async def test_read_file_still_returns_a_text_file_whole(tmp_path):
    from tools.file_manager import ReadFileTool

    note = tmp_path / "note.txt"
    note.write_text("line one\nline two\n")
    result = await ReadFileTool().run(str(note))
    assert result.success and result.output == "line one\nline two\n"


@pytest.mark.asyncio
async def test_a_text_file_too_big_to_return_whole_returns_its_start(tmp_path):
    from tools.file_manager import ReadFileTool

    big = tmp_path / "huge.txt"
    big.write_text("\n".join(f"line {i}" for i in range(120_000)))
    result = await ReadFileTool().run(str(big))
    assert result.success
    assert result.output["characters"] <= documents.DEFAULT_EXTRACT_CHARS
    assert "read_document" in result.output["note"]
