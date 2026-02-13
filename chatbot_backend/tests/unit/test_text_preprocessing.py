from app.services.ingestion.text_preprocessing import text_processing, extract_json


def test_text_processing_removes_blank_lines():
    raw = "Hello\n\n\n\nWorld"
    result = text_processing(raw)
    assert "\n\n\n" not in result
    assert "Hello" in result
    assert "World" in result


def test_text_processing_removes_page_headers():
    raw = "Page 1 of 3\nContent here"
    result = text_processing(raw)
    assert "Page 1 of 3" not in result
    assert "Content here" in result


def test_extract_json_valid():
    text = 'Some text {"key": "value"} more text'
    result = extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_invalid():
    text = "no json here"
    result = extract_json(text)
    assert result is None
