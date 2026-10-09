import json
from server import (
    ocr_init_task, ocr_extract_text, ocr_estimate_width, ocr_get_bboxes,
    InitArgs, ExtractArgs, HandleArgs, ResponseFormat
)

def test_ocr_init_success():
    args = InitArgs(file_path="scan.jpg")
    res = ocr_init_task(args)
    assert res.is_error is False
    assert "doc_42" in res.content[0].text

def test_ocr_init_error():
    args = InitArgs(file_path="video.mp4")
    res = ocr_init_task(args)
    assert res.is_error is True
    assert "неподдерживаемый формат" in res.content[0].text

def test_ocr_extract_text_concise():
    args = ExtractArgs(handle="doc_42", response_format=ResponseFormat.CONCISE)
    res = ocr_extract_text(args)
    assert res.is_error is False
    data = json.loads(res.content[0].text)
    assert "metadata" not in data
    assert data["text"] == "Сумма: 1000 руб."

def test_ocr_extract_text_error():
    args = ExtractArgs(handle="invalid_doc", response_format=ResponseFormat.DETAILED)
    res = ocr_extract_text(args)
    assert res.is_error is True
    assert "вызовите ocr_init_task" in res.content[0].text

def test_ocr_estimate_width_success():
    args = HandleArgs(handle="doc_42")
    res = ocr_estimate_width(args)
    assert res.is_error is False
    assert "12.5" in res.content[0].text

def test_ocr_get_bboxes_error():
    args = HandleArgs(handle="bad_handle")
    res = ocr_get_bboxes(args)
    assert res.is_error is True
    assert "Смените handle" in res.content[0].text