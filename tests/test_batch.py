"""CSV import uses synthetic text only; no model is required."""
import importlib
import json
import threading
import unittest
import urllib.error
import urllib.request


class BatchTests(unittest.TestCase):
    def parse(self, text):
        return importlib.import_module('avito.batch').parse_csv(text.encode('utf-8'))

    def test_delimiters_bom_quotes_and_display_title(self):
        for separator in (',', ';', '\t'):
            with self.subTest(separator=separator):
                text = '\ufeffid' + separator + 'title' + separator + 'description\r\n'
                text += ' 007 ' + separator + 'Заголовок' + separator + '"Плитка, швы; пол\nВанная"\r\n'
                self.assertEqual(self.parse(text), [{'id': '007', 'title': 'Заголовок',
                                                    'description': 'Плитка, швы; пол\nВанная'}])
        self.assertEqual(self.parse('description\nРабота\nЕще работа\n'), [
            {'id': '1', 'title': '', 'description': 'Работа'},
            {'id': '2', 'title': '', 'description': 'Еще работа'}])

    def test_rejects_ambiguous_or_malformed_rows(self):
        invalid = ('', 'title\nТекст', 'description,description\na,b',
                   'description,unknown\na,b', 'description,title\na',
                   'description\na,b', 'description\n"не закрыто',
                   'description\n"текст"мусор', 'description\n   ',
                   'id,description\nx,a\nx,b', 'id,description\n,a',
                   'description\n', 'description\n' + 'a' * 12001)
        for text in invalid:
            with self.subTest(text=text[:70]):
                with self.assertRaises(ValueError):
                    self.parse(text)

    def test_record_byte_and_encoding_limits(self):
        self.assertEqual(len(self.parse('description\n' + 'Текст\n' * 200)), 200)
        with self.assertRaises(ValueError):
            self.parse('description\n' + 'Текст\n' * 201)
        parse_csv = importlib.import_module('avito.batch').parse_csv
        for payload in (b'description\n\xff', b'a' * (2 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                parse_csv(payload)

    def test_official_testcase_alias_preserves_ids_and_rejects_ambiguity(self):
        self.assertEqual(self.parse('testcase,description\ncase-001,Синтетическая услуга\n'),
                         [{'id': 'case-001', 'title': '', 'description': 'Синтетическая услуга'}])
        with self.assertRaises(ValueError):
            self.parse('id,testcase,description\na,b,Синтетическая услуга\n')


class BatchHTTPTests(unittest.TestCase):
    def test_import_limits_errors_and_same_origin(self):
        from avito.web import make_server
        with make_server(0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def request(path, body, content_type='text/csv', origin=None):
                headers = {'Content-Type': content_type}
                if origin:
                    headers['Origin'] = origin
                req = urllib.request.Request(base + path, data=body, headers=headers)
                try:
                    response = urllib.request.urlopen(req, timeout=10)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    return response.status, json.load(response)
            try:
                status, data = request('/import-csv', b'description\n' + b'a' * 12000 + b'\n' * 70000)
                self.assertEqual(status, 200)
                self.assertEqual(data['items'][0]['id'], '1')
                for body in (b'description\n\xff', b'description\n"unfinished'):
                    status, data = request('/import-csv', body)
                    self.assertEqual(status, 422)
                    self.assertIn('error', data)
                self.assertEqual(request('/import-csv', b'description\na', origin='https://evil.test')[0], 403)
                self.assertEqual(request('/import-csv', b'description\na', 'application/json')[0], 400)
                self.assertEqual(request('/import-csv', b'x' * (2 * 1024 * 1024 + 1))[0], 413)
                self.assertEqual(request('/detect', b'x' * 65537, 'application/json')[0], 413)
            finally:
                server.shutdown()
                thread.join()
