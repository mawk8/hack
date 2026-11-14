"""Loopback-only demo API on Python's standard library."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import threading

from .llm import (AnalysisError, DEFAULT_ENDPOINT, DEFAULT_MODEL, analyze,
                  parse_json, validate_endpoint, validate_input)

MAX_BODY_BYTES = 65536


def make_server(port=7860, endpoint=DEFAULT_ENDPOINT, model=DEFAULT_MODEL):
    endpoint = validate_endpoint(endpoint)
    ui = Path(__file__).with_name('ui.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        # No advertisements or request headers are logged.
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_bytes(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode('utf-8'))

        def fail(self, status, code, message, elapsed=0):
            self.close_connection = True
            self.send_json(status, {'error': {'code': code, 'message': message,
                                             'elapsed_seconds': elapsed}})

        def do_GET(self):
            if self.path == '/':
                self.send_bytes(200, ui, 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.send_json(200, {'status': 'ok', 'model_status': 'not_checked'})
            else:
                self.fail(404, 'not_found', 'Путь не найден.')

        def do_POST(self):
            if self.path not in ('/predict', '/analyze'):
                self.fail(404, 'not_found', 'Путь не найден.')
                return
            # JSON + same-origin browser requests prevent cross-site GPU calls.
            origin = self.headers.get('Origin')
            if origin is not None and origin != f'http://127.0.0.1:{self.server.server_port}':
                self.fail(403, 'origin_rejected', 'Разрешены запросы с локальной страницы приложения.')
                return
            if self.headers.get('Transfer-Encoding'):
                self.fail(400, 'invalid_body', 'Transfer-Encoding не поддерживается.')
                return
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                self.fail(400, 'invalid_body', 'Нужен один корректный Content-Length.')
                return
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY_BYTES:
                self.fail(413, 'body_too_large', 'Допустим непустой JSON размером до 65536 байт.')
                return
            if self.headers.get_content_type() != 'application/json':
                self.fail(400, 'invalid_body', 'Нужен Content-Type: application/json.')
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    self.fail(400, 'invalid_body', 'Тело запроса получено не полностью.')
                    return
                data = parse_json(raw)
            except (ValueError, UnicodeError, RecursionError):
                self.fail(400, 'invalid_json', 'Неверный JSON запроса.')
                return
            except (socket.timeout, OSError):
                self.fail(400, 'invalid_body', 'Не удалось прочитать тело запроса.')
                return
            if type(data) is not dict or set(data) != {'description', 'mcId'}:
                self.fail(422, 'invalid_input', 'Нужны ровно поля description и mcId.')
                return
            try:
                validate_input(data['description'], data['mcId'])
            except AnalysisError as error:
                self.fail(error.status, error.code, str(error))
                return
            if not self.server.inference_lock.acquire(blocking=False):
                self.fail(503, 'busy', 'Модель занята другим запросом. Повторите позже.')
                return
            try:
                output = analyze(data['description'], data['mcId'], endpoint=endpoint, model=model)
                self.send_json(200, output['result'] if self.path == '/predict' else output)
            except AnalysisError as error:
                self.fail(error.status, error.code, str(error), error.elapsed_seconds)
            except Exception:
                self.fail(500, 'internal_error', 'Внутренняя ошибка приложения.')
            finally:
                self.server.inference_lock.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    # ponytail: one GPU, one inference; use a queue only if concurrent service is needed.
    server.inference_lock = threading.Lock()
    return server


def serve(port=7860, endpoint=DEFAULT_ENDPOINT, model=DEFAULT_MODEL):
    with make_server(port, endpoint, model) as server:
        print(f'Демо: http://127.0.0.1:{server.server_port} (модель: {endpoint})', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
