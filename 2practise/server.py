import json
from enum import Enum
from pydantic import BaseModel, Field
import mcp.types as types
from mcp.server import MCPServer

# Инициализация сервера
server = MCPServer("ocr-tools")

# Имитация хранилища состояний. 
# Протокол 2026-07-28 работает без сессий, поэтому состояние вызывается по handle[cite: 27, 29].
DOC_STATE = {
    "doc_42": {"status": "ready", "text": "Сумма: 1000 руб.", "width": 12.5, "bboxes": 45}
}

class ResponseFormat(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"

class BaseToolArgs(BaseModel):
    # Гарантирует additionalProperties: false для закрытых множеств[cite: 10, 65]
    model_config = {"extra": "forbid"}

class InitArgs(BaseToolArgs):
    file_path: str = Field(..., description="Путь к скану документа. Используйте для инициализации. Не используйте для видео и аудио.")

class HandleArgs(BaseToolArgs):
    handle: str = Field(..., description="Идентификатор документа (handle), полученный из ocr_init_task.")

class ExtractArgs(HandleArgs):
    response_format: ResponseFormat = Field(..., description="Формат ответа: concise (только текст) или detailed (текст с метаданными).")

def create_error(hint: str) -> types.CallToolResult:
    """Вспомогательная функция для возврата ошибки как наблюдения[cite: 11]."""
    return types.CallToolResult(
        is_error=True,
        content=[types.TextContent(type="text", text=hint)]
    )

def create_success(data: dict) -> types.CallToolResult:
    """Вспомогательная функция для успешного ответа."""
    return types.CallToolResult(
        is_error=False,
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
    )

@server.tool()
def ocr_init_task(args: InitArgs) -> types.CallToolResult:
    """Создает задачу на OCR для переданного файла."""
    if not args.file_path.endswith((".jpg", ".png", ".pdf")):
        return create_error("Ошибка: неподдерживаемый формат. Пожалуйста, передайте файл с расширением .jpg, .png или .pdf.")
    return create_success({"handle": "doc_42"})

@server.tool()
def ocr_estimate_width(args: HandleArgs) -> types.CallToolResult:
    """Оценивает среднюю ширину символов в распознанном документе."""
    if args.handle not in DOC_STATE:
        return create_error(f"Ошибка: handle '{args.handle}' не найден или истёк. Вызовите ocr_init_task заново.")
    return create_success({"width": DOC_STATE[args.handle]["width"]})

@server.tool()
def ocr_extract_text(args: ExtractArgs) -> types.CallToolResult:
    """Возвращает распознанный текст документа."""
    if args.handle not in DOC_STATE:
        return create_error("Ошибка: документ не найден. Сначала вызовите ocr_init_task.")
    
    data = DOC_STATE[args.handle]
    if args.response_format == ResponseFormat.CONCISE:
        return create_success({"text": data["text"]})
    return create_success({"text": data["text"], "metadata": {"status": data["status"], "bboxes": data["bboxes"]}})

@server.tool()
def ocr_get_bboxes(args: HandleArgs) -> types.CallToolResult:
    """Возвращает количество найденных bounding boxes."""
    if args.handle not in DOC_STATE:
        return create_error("Ошибка: документ не найден. Смените handle или загрузите файл заново.")
    return create_success({"bboxes": DOC_STATE[args.handle]["bboxes"]})

if __name__ == "__main__":
    # Транспорт stdio для интеграции с клиентами[cite: 31, 32]
    server.run(transport="stdio")