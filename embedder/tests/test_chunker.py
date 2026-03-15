"""Comprehensive tests for chunker module.

Tests exercise the production entry point (chunk_file) across every file type,
plus edge cases discovered during real indexing runs:
- 97MB XML file hanging the chunker (LARGE_FILE_CHARS fast path)
- RecursiveJsonSplitter convert_lists TypeError (_make_json_splitter compat)
- Empty string chunks sent to embedding backend (no-empty-content invariant)
"""

import json
from pathlib import Path

import pytest

from opencode_embedder.chunker import (
    LARGE_FILE_CHARS,
    Chunk,
    chunk_file,
    count_tokens,
    set_tier,
    split_by_tokens,
    _make_json_splitter,
)


@pytest.fixture(autouse=True)
def setup_tier():
    """Set tier for tests"""
    set_tier("premium")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_chunks(chunks: list[Chunk], *, min_count: int = 1):
    """Assert every chunk has valid metadata and non-empty content."""
    assert len(chunks) >= min_count
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.content.strip(), f"Chunk has empty content: {chunk!r}"
        assert chunk.start_line >= 0
        assert chunk.end_line >= chunk.start_line
        assert chunk.language


def _large_content(pattern: str, target_chars: int) -> str:
    """Repeat a pattern until it exceeds target_chars."""
    repeats = (target_chars // len(pattern)) + 1
    return (pattern * repeats)[: target_chars + 100]


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------


class TestBasic:
    def test_count_tokens(self):
        assert count_tokens("Hello, world!") > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_small_content_single_chunk(self):
        chunks = chunk_file("x = 1", Path("t.py"))
        _assert_valid_chunks(chunks, min_count=1)
        assert len(chunks) == 1

    def test_empty_content(self):
        chunks = chunk_file("", Path("t.py"))
        assert len(chunks) == 0  # empty file returns empty list


# ---------------------------------------------------------------------------
# Large file fast path (LARGE_FILE_CHARS)
# ---------------------------------------------------------------------------


class TestLargeFileFastPath:
    """Files > LARGE_FILE_CHARS skip structure-aware parsing entirely.

    This prevents the 97MB XML file hang scenario.
    """

    def test_large_xml_uses_fallback(self):
        """97MB XML must not hang — uses fast TokenChunker fallback."""
        content = _large_content(
            "<item><name>test</name><value>123</value></item>\n",
            LARGE_FILE_CHARS + 1000,
        )
        chunks = chunk_file(content, Path("big.xml"))
        _assert_valid_chunks(chunks)
        # Should produce multiple chunks (not a single giant one)
        assert len(chunks) > 1

    def test_large_json_uses_fallback(self):
        content = _large_content('{"key": "value", "n": 42}\n', LARGE_FILE_CHARS + 1000)
        chunks = chunk_file(content, Path("big.json"))
        _assert_valid_chunks(chunks)

    def test_large_html_uses_fallback(self):
        content = _large_content(
            "<div><p>Some paragraph content here.</p></div>\n",
            LARGE_FILE_CHARS + 1000,
        )
        chunks = chunk_file(content, Path("big.html"))
        _assert_valid_chunks(chunks)

    def test_large_markdown_uses_fallback(self):
        content = _large_content("## Section\n\nParagraph text here.\n\n", LARGE_FILE_CHARS + 1000)
        chunks = chunk_file(content, Path("big.md"))
        _assert_valid_chunks(chunks)

    def test_large_python_uses_fallback(self):
        content = _large_content("def fn():\n    pass\n\n", LARGE_FILE_CHARS + 1000)
        chunks = chunk_file(content, Path("big.py"))
        _assert_valid_chunks(chunks)

    def test_below_threshold_uses_router(self):
        """Content just under LARGE_FILE_CHARS should use the proper router."""
        # Generate markdown under the limit but over TARGET_TOKENS
        content = "# Title\n\n" + ("word " * 5000) + "\n\n## Section\n\n" + ("text " * 5000)
        assert len(content) < LARGE_FILE_CHARS
        chunks = chunk_file(content, Path("normal.md"))
        _assert_valid_chunks(chunks)
        # Markdown router should split on headers
        assert any("markdown" == c.chunk_type for c in chunks)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJSON:
    def test_simple_object(self):
        data = {"key": "value", "nested": {"inner": "data"}}
        chunks = chunk_file(json.dumps(data), Path("test.json"))
        _assert_valid_chunks(chunks)

    def test_large_nested(self):
        """Deeply nested JSON gets split into multiple chunks."""
        data = {"level0": {f"key{i}": {"data": "x" * 500} for i in range(100)}}
        content = json.dumps(data, indent=2)
        assert len(content) > 16_000  # exceeds TARGET_CHARS
        chunks = chunk_file(content, Path("nested.json"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1

    def test_invalid_json_fallback(self):
        """Invalid JSON falls back to TokenChunker without crashing."""
        content = "{not valid json" + " data" * 5000
        chunks = chunk_file(content, Path("bad.json"))
        _assert_valid_chunks(chunks)

    def test_array_root(self):
        data = [{"id": i, "value": f"item_{i}" * 100} for i in range(50)]
        chunks = chunk_file(json.dumps(data, indent=2), Path("array.json"))
        _assert_valid_chunks(chunks)

    def test_jsonc_extension(self):
        """JSONC extension routes to JSON chunker."""
        chunks = chunk_file('{"key": "value"}', Path("config.jsonc"))
        _assert_valid_chunks(chunks)

    def test_json5_extension(self):
        """JSON5 extension routes to JSON chunker (may fallback on parse error)."""
        chunks = chunk_file('{"key": "value"}', Path("config.json5"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# _make_json_splitter (convert_lists compatibility fix)
# ---------------------------------------------------------------------------


class TestMakeJsonSplitter:
    def test_returns_working_splitter(self):
        """_make_json_splitter returns a splitter regardless of convert_lists support."""
        splitter = _make_json_splitter()
        assert splitter is not None
        # Should be able to split a simple dict
        texts = splitter.split_text(json_data={"key": "value"})
        assert len(texts) >= 1

    def test_splits_nested_data(self):
        splitter = _make_json_splitter()
        data = {f"k{i}": {"nested": "x" * 200} for i in range(20)}
        texts = splitter.split_text(json_data=data)
        assert len(texts) >= 1
        for t in texts:
            assert isinstance(t, str)


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


class TestJSONL:
    def test_simple_lines(self):
        lines = [json.dumps({"id": i, "text": f"item {i}"}) for i in range(10)]
        chunks = chunk_file("\n".join(lines), Path("data.jsonl"))
        _assert_valid_chunks(chunks)

    def test_empty_lines_ignored(self):
        content = '{"a":1}\n\n\n{"b":2}\n\n{"c":3}'
        chunks = chunk_file(content, Path("sparse.jsonl"))
        _assert_valid_chunks(chunks)

    def test_large_jsonl_splits(self):
        lines = [json.dumps({"id": i, "payload": "x" * 500}) for i in range(200)]
        content = "\n".join(lines)
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("big.jsonl"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


class TestYAML:
    def test_simple(self):
        content = "key: value\nnested:\n  inner: data\nlist:\n  - item1\n  - item2\n"
        chunks = chunk_file(content, Path("test.yaml"))
        _assert_valid_chunks(chunks)

    def test_yml_extension(self):
        chunks = chunk_file("key: value\n", Path("test.yml"))
        _assert_valid_chunks(chunks)

    def test_multi_document(self):
        content = "---\na: 1\n---\nb: 2\n---\nc: 3\n"
        chunks = chunk_file(content, Path("multi.yaml"))
        _assert_valid_chunks(chunks)

    def test_empty_documents_filtered(self):
        """YAML with None documents (---\\n---) doesn't crash."""
        content = "---\n---\n---\nkey: value\n"
        chunks = chunk_file(content, Path("empty_docs.yaml"))
        _assert_valid_chunks(chunks)

    def test_scalar_value_fallback(self):
        """YAML that parses to a scalar (not dict/list) returns single chunk."""
        content = "just a plain string value\n" * 100
        chunks = chunk_file(content, Path("scalar.yaml"))
        _assert_valid_chunks(chunks)

    def test_large_yaml(self):
        entries = {
            f"service_{i}": {"host": "localhost", "port": 8000 + i, "config": {"key": "x" * 200}}
            for i in range(100)
        }
        import yaml

        content = yaml.dump(entries, default_flow_style=False)
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("big.yaml"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------


class TestTOML:
    def test_simple(self):
        content = '[package]\nname = "test"\nversion = "1.0"\n\n[dependencies]\nfoo = "1.2"\n'
        chunks = chunk_file(content, Path("Cargo.toml"))
        _assert_valid_chunks(chunks)

    def test_invalid_toml_fallback(self):
        """Invalid TOML falls back gracefully."""
        content = "not = valid [ toml" + " data" * 5000
        chunks = chunk_file(content, Path("bad.toml"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


class TestHTML:
    def test_simple_page(self):
        content = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        chunks = chunk_file(content, Path("page.html"))
        _assert_valid_chunks(chunks)

    def test_with_table(self):
        rows = "".join(f"<tr><td>Row {i}</td><td>Data {i}</td></tr>" for i in range(50))
        content = f"<html><body><h1>Report</h1><table>{rows}</table></body></html>"
        chunks = chunk_file(content, Path("report.html"))
        _assert_valid_chunks(chunks)

    def test_htm_extension(self):
        chunks = chunk_file("<p>Hello</p>", Path("page.htm"))
        _assert_valid_chunks(chunks)

    def test_xhtml_extension(self):
        chunks = chunk_file("<p>Hello</p>", Path("page.xhtml"))
        _assert_valid_chunks(chunks)

    def test_large_html_with_lists(self):
        """Large HTML with preserved <ul> may stay as one chunk (splitter preserves lists)."""
        items = "".join(f"<li>Item {i}: {'x' * 200}</li>" for i in range(200))
        content = f"<html><body><h1>Catalog</h1><ul>{items}</ul></body></html>"
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("catalog.html"))
        _assert_valid_chunks(chunks)

    def test_large_html_multiple_sections(self):
        """HTML with multiple header sections splits into multiple chunks."""
        sections = "".join(f"<h2>Section {i}</h2><p>{'Content ' * 500}</p>" for i in range(20))
        content = f"<html><body><h1>Title</h1>{sections}</body></html>"
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("multi.html"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------


class TestXML:
    def test_simple_xml(self):
        content = '<?xml version="1.0"?>\n<root><item>Hello</item><item>World</item></root>'
        chunks = chunk_file(content, Path("data.xml"))
        _assert_valid_chunks(chunks)

    def test_xsl_extension(self):
        chunks = chunk_file("<xsl:stylesheet/>", Path("style.xsl"))
        _assert_valid_chunks(chunks)

    def test_plist_extension(self):
        content = '<?xml version="1.0"?>\n<plist><dict><key>Name</key><string>Test</string></dict></plist>'
        chunks = chunk_file(content, Path("Info.plist"))
        _assert_valid_chunks(chunks)

    def test_medium_xml_uses_router(self):
        """XML under LARGE_FILE_CHARS uses structure-aware splitter."""
        items = "".join(f"<item><id>{i}</id><data>{'x' * 200}</data></item>\n" for i in range(200))
        content = f'<?xml version="1.0"?>\n<root>\n{items}</root>'
        assert len(content) < LARGE_FILE_CHARS
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("medium.xml"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


class TestLaTeX:
    def test_simple_document(self):
        content = (
            "\\documentclass{article}\n\\begin{document}\n"
            "\\section{Introduction}\nHello world.\n"
            "\\section{Methods}\nSome methods.\n"
            "\\end{document}\n"
        )
        chunks = chunk_file(content, Path("paper.tex"))
        _assert_valid_chunks(chunks)

    def test_latex_extension(self):
        chunks = chunk_file("\\section{A}\nContent.\n", Path("doc.latex"))
        _assert_valid_chunks(chunks)

    def test_ltx_extension(self):
        chunks = chunk_file("\\section{A}\nContent.\n", Path("doc.ltx"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# RST
# ---------------------------------------------------------------------------


class TestRST:
    def test_simple_document(self):
        content = (
            "Title\n=====\n\nSome introduction text.\n\nSection\n-------\n\nMore details here.\n"
        )
        chunks = chunk_file(content, Path("readme.rst"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


class TestMarkdown:
    def test_header_splitting(self):
        content = "# Title\n\nIntro.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.\n"
        chunks = chunk_file(content, Path("doc.md"))
        _assert_valid_chunks(chunks)

    def test_mdx_extension(self):
        chunks = chunk_file("# Hello\n\nWorld.\n", Path("page.mdx"))
        _assert_valid_chunks(chunks)

    def test_large_markdown_splits(self):
        sections = "\n\n".join(f"## Section {i}\n\n{'Content ' * 500}" for i in range(20))
        content = f"# Document\n\n{sections}"
        assert len(content) > 16_000
        chunks = chunk_file(content, Path("big.md"))
        _assert_valid_chunks(chunks)
        assert len(chunks) > 1
        assert all(c.chunk_type == "markdown" for c in chunks)

    def test_code_blocks_preserved(self):
        content = "# API\n\n```python\ndef hello():\n    print('hi')\n```\n\nDescription.\n"
        chunks = chunk_file(content, Path("api.md"))
        _assert_valid_chunks(chunks)
        combined = " ".join(c.content for c in chunks)
        assert "def hello" in combined


# ---------------------------------------------------------------------------
# Code (tree-sitter AST)
# ---------------------------------------------------------------------------


class TestCode:
    def test_python(self):
        content = "\n\n".join(f"def fn_{i}():\n    return {i}" for i in range(20))
        chunks = chunk_file(content, Path("module.py"))
        _assert_valid_chunks(chunks)

    def test_go(self):
        funcs = "\n\n".join(f"func Fn{i}() int {{\n\treturn {i}\n}}" for i in range(20))
        content = f"package main\n\n{funcs}\n"
        chunks = chunk_file(content, Path("main.go"))
        _assert_valid_chunks(chunks)

    def test_typescript(self):
        funcs = "\n\n".join(
            f"export function fn{i}(): number {{\n  return {i};\n}}" for i in range(20)
        )
        content = funcs
        chunks = chunk_file(content, Path("utils.ts"))
        _assert_valid_chunks(chunks)

    def test_tsx_extension(self):
        content = "export function App() {\n  return <div>Hello</div>;\n}\n"
        chunks = chunk_file(content, Path("App.tsx"))
        _assert_valid_chunks(chunks)

    def test_rust(self):
        content = 'fn main() {\n    println!("Hello");\n}\n\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n'
        chunks = chunk_file(content, Path("main.rs"))
        _assert_valid_chunks(chunks)

    def test_java(self):
        content = 'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}\n'
        chunks = chunk_file(content, Path("Main.java"))
        _assert_valid_chunks(chunks)

    def test_sql(self):
        content = "SELECT * FROM users WHERE id = 1;\nINSERT INTO logs VALUES (1, 'test');\n"
        chunks = chunk_file(content, Path("query.sql"))
        _assert_valid_chunks(chunks)

    def test_dockerfile(self):
        content = 'FROM python:3.12\nRUN pip install flask\nCOPY . /app\nCMD ["python", "app.py"]\n'
        chunks = chunk_file(content, Path("Dockerfile"))
        _assert_valid_chunks(chunks)

    def test_css(self):
        content = "body { margin: 0; }\n.container { max-width: 1200px; }\n"
        chunks = chunk_file(content, Path("style.css"))
        _assert_valid_chunks(chunks)

    def test_unknown_language_fallback(self):
        """Unknown extension falls back to prose or token chunker."""
        content = "Some random content\n" * 500
        chunks = chunk_file(content, Path("data.xyz"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# Prose / text
# ---------------------------------------------------------------------------


class TestProse:
    def test_plain_text(self):
        paragraphs = "\n\n".join(f"Paragraph {i}. " + "Word " * 200 for i in range(20))
        chunks = chunk_file(paragraphs, Path("essay.txt"))
        _assert_valid_chunks(chunks)

    def test_text_semantic_chunking(self):
        """Text files use SemanticChunker, grouping related content."""
        content = (
            "Machine learning is a subset of artificial intelligence. "
            "It allows systems to learn from data. "
            "Neural networks are a key technique. " * 50 + "\n\n"
            "Cooking requires fresh ingredients. "
            "The recipe calls for butter and flour. "
            "Preheat the oven to 350 degrees. " * 50
        )
        chunks = chunk_file(content, Path("mixed.txt"))
        _assert_valid_chunks(chunks)


# ---------------------------------------------------------------------------
# No empty chunks invariant
# ---------------------------------------------------------------------------


class TestNoEmptyChunks:
    """Every chunk produced by chunk_file must have non-empty stripped content.

    This prevents embedding backends from rejecting inputs with empty strings.
    """

    def _assert_no_empty(self, chunks: list[Chunk]):
        for i, chunk in enumerate(chunks):
            assert chunk.content.strip(), f"Chunk {i} has empty content"

    def test_xml_no_empty_chunks(self):
        items = "".join(f"<item>{i}</item>\n" for i in range(500))
        content = f"<root>\n{items}</root>"
        self._assert_no_empty(chunk_file(content, Path("data.xml")))

    def test_json_no_empty_chunks(self):
        data = {f"key_{i}": {"value": i, "text": f"data {i}"} for i in range(100)}
        self._assert_no_empty(chunk_file(json.dumps(data, indent=2), Path("data.json")))

    def test_html_no_empty_chunks(self):
        content = "<html><body>" + "<p>Text</p>\n" * 500 + "</body></html>"
        self._assert_no_empty(chunk_file(content, Path("page.html")))

    def test_markdown_no_empty_chunks(self):
        content = "\n\n".join(f"## Heading {i}\n\n{'Word ' * 200}" for i in range(30))
        self._assert_no_empty(chunk_file(content, Path("doc.md")))

    def test_yaml_no_empty_chunks(self):
        entries = {f"svc_{i}": {"port": 8000 + i} for i in range(100)}
        import yaml

        self._assert_no_empty(chunk_file(yaml.dump(entries), Path("config.yaml")))

    def test_python_no_empty_chunks(self):
        content = "\n\n".join(f"def fn_{i}():\n    return {i}" for i in range(100))
        self._assert_no_empty(chunk_file(content, Path("module.py")))

    def test_whitespace_only_content(self):
        """Content that is all whitespace returns empty list."""
        chunks = chunk_file("   \n\n   \n   ", Path("blank.py"))
        assert len(chunks) == 0  # whitespace-only returns empty list

    def test_mixed_empty_lines(self):
        """Content with many empty lines between real content."""
        content = "\n\n\n".join(f"Line {i}" for i in range(200))
        chunks = chunk_file(content, Path("sparse.txt"))
        self._assert_no_empty(chunks)


# ---------------------------------------------------------------------------
# split_by_tokens (used by _enforce_token_limit)
# ---------------------------------------------------------------------------


class TestSplitByTokens:
    def test_empty_string(self):
        assert split_by_tokens("", "text") == []

    def test_small_content(self):
        chunks = split_by_tokens("Hello world", "text")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"

    def test_whitespace_only_skipped(self):
        """Whitespace-only segments produce no chunks."""
        chunks = split_by_tokens("   \n\n   ", "text")
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_chunk_has_language(self):
        chunks = chunk_file("x = 1\ny = 2\n", Path("test.py"))
        for c in chunks:
            assert c.language

    def test_chunk_has_type(self):
        chunks = chunk_file("# Title\n\nContent.\n", Path("doc.md"))
        for c in chunks:
            assert c.chunk_type

    def test_chunk_line_numbers(self):
        chunks = chunk_file("line1\nline2\nline3\n", Path("test.py"))
        for c in chunks:
            assert c.start_line >= 0
            assert c.end_line >= 0

    def test_markdown_chunk_type(self):
        content = "# Title\n\n" + "Word " * 5000 + "\n\n## Section\n\n" + "Text " * 5000
        chunks = chunk_file(content, Path("doc.md"))
        assert all(c.chunk_type == "markdown" for c in chunks)


# ---------------------------------------------------------------------------
# Extension routing coverage
# ---------------------------------------------------------------------------


class TestExtensionRouting:
    """Verify each supported extension hits the correct chunker path."""

    @pytest.mark.parametrize("ext", ["md", "mdx"])
    def test_markdown_extensions(self, ext):
        content = "# Title\n\nContent.\n"
        chunks = chunk_file(content, Path(f"doc.{ext}"))
        _assert_valid_chunks(chunks)

    @pytest.mark.parametrize("ext", ["json", "jsonc", "json5"])
    def test_json_extensions(self, ext):
        chunks = chunk_file('{"key": "value"}', Path(f"data.{ext}"))
        _assert_valid_chunks(chunks)

    @pytest.mark.parametrize("ext", ["yaml", "yml"])
    def test_yaml_extensions(self, ext):
        chunks = chunk_file("key: value\n", Path(f"config.{ext}"))
        _assert_valid_chunks(chunks)

    @pytest.mark.parametrize("ext", ["html", "htm", "xhtml"])
    def test_html_extensions(self, ext):
        chunks = chunk_file("<p>Hello</p>", Path(f"page.{ext}"))
        _assert_valid_chunks(chunks)

    @pytest.mark.parametrize("ext", ["xml", "xsl", "xslt", "plist"])
    def test_xml_extensions(self, ext):
        chunks = chunk_file("<root/>", Path(f"file.{ext}"))
        _assert_valid_chunks(chunks)

    @pytest.mark.parametrize("ext", ["tex", "latex", "ltx"])
    def test_latex_extensions(self, ext):
        chunks = chunk_file("\\section{A}\nText.\n", Path(f"doc.{ext}"))
        _assert_valid_chunks(chunks)

    def test_rst_extension(self):
        chunks = chunk_file("Title\n=====\n\nText.\n", Path("doc.rst"))
        _assert_valid_chunks(chunks)

    def test_toml_extension(self):
        chunks = chunk_file('[pkg]\nname = "x"\n', Path("cfg.toml"))
        _assert_valid_chunks(chunks)

    def test_jsonl_extension(self):
        chunks = chunk_file('{"a":1}\n{"b":2}\n', Path("data.jsonl"))
        _assert_valid_chunks(chunks)
