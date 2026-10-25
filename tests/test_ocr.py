from app.ocr import parse_result


def test_parse_result_with_boxes() -> None:
    result = {
        "rec_texts": ["Hello", "World"],
        "rec_scores": [0.9, 1.2],
        "rec_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
    }
    lines = parse_result(result)
    assert len(lines) == 2
    assert lines[0].text == "Hello"
    assert lines[0].box == [1.0, 2.0, 3.0, 4.0]
    assert lines[1].confidence == 1.0


def test_parse_result_falls_back_to_polygons() -> None:
    result = {
        "rec_texts": ["Line"],
        "rec_scores": [0.75],
        "rec_polys": [[[10, 20], [110, 20], [110, 60], [10, 60]]],
    }
    lines = parse_result(result)
    assert lines[0].box == [10.0, 20.0, 110.0, 60.0]


def test_parse_result_skips_empty_text() -> None:
    result = {"rec_texts": ["", "Kept"], "rec_scores": [0.1, 0.2]}
    lines = parse_result(result)
    assert [line.text for line in lines] == ["Kept"]


def test_parse_result_handles_missing_fields() -> None:
    assert parse_result({}) == []
