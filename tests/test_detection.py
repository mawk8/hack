"""Synthetic transport/schema checks, not semantic LLM accuracy."""
import json
import unittest

from avito import llm
from tests import test_llm as fixtures
from tests.test_llm import SOURCE, SERVICE, fake_model

CATEGORY = {key: SERVICE[key] for key in ('mcId', 'evidence', 'constraints')}


def detection_model(categories=None, **kwargs):
    return fake_model(content=json.dumps({'categories': [CATEGORY] if categories is None else categories}),
                      **kwargs)


class DetectionTests(unittest.TestCase):
    def test_detect_only_categories_and_separate_prompt(self):
        with detection_model() as (endpoint, calls):
            output = llm.detect(SOURCE, 101, endpoint=endpoint)
        self.assertEqual(set(output), {'categories', 'meta'})
        self.assertEqual(output['categories'], [CATEGORY])
        self.assertEqual(output['meta']['attempts'], 1)
        self.assertEqual(output['meta']['usage']['total_tokens'], 73)
        prompt = calls[0]['body']['messages'][0]['content']
        for field in ('"title"', '"text"', '"services"'):
            self.assertNotIn(field, prompt)
        self.assertEqual(json.loads(calls[0]['body']['messages'][1]['content']),
                         {'description': SOURCE, 'mcId': 101})

    def test_detect_validates_schema_source_duplicates_and_quotes(self):
        cases = [[CATEGORY, CATEGORY], [dict(CATEGORY, mcId=101)],
                 [dict(CATEGORY, mcId=True)], [dict(CATEGORY, mcId=999)],
                 [dict(CATEGORY, evidence=[])], [dict(CATEGORY, evidence=['absent'])],
                 [dict(CATEGORY, constraints=['absent'])],
                 [dict(CATEGORY, title='unexpected')]]
        for categories in cases:
            with self.subTest(categories=categories), detection_model(categories) as (endpoint, _):
                with self.assertRaises(llm.ModelError):
                    llm.detect(SOURCE, 101, endpoint=endpoint)
        with detection_model([]) as (endpoint, _):
            self.assertEqual(llm.detect(SOURCE, 101, endpoint=endpoint)['categories'], [])

    def test_detect_malformed_incomplete_and_transport_errors(self):
        for content in ('{}', '{"categories":', '{"categories":[],"categories":[]}',
                        '{"categories":null}', '{"services":[]}'):
            with self.subTest(content=content), fake_model(content=content) as (endpoint, _):
                with self.assertRaises(llm.ModelError):
                    llm.detect(SOURCE, 101, endpoint=endpoint)
        with detection_model(finish_reason='length') as (endpoint, _):
            with self.assertRaises(llm.ModelError):
                llm.detect(SOURCE, 101, endpoint=endpoint)
        with detection_model(delay=.1) as (endpoint, _):
            with self.assertRaises(llm.ModelTimeout):
                llm.detect(SOURCE, 101, endpoint=endpoint, timeout=.01)
        with detection_model(status=500) as (endpoint, _):
            with self.assertRaises(llm.ModelError):
                llm.detect(SOURCE, 101, endpoint=endpoint)

    def test_generate_exact_selection_and_full_source_without_server_state(self):
        description = SOURCE + ' Электрика.'
        with fake_model() as (endpoint, calls):
            output = llm.generate(description, 101, [103], endpoint=endpoint)
        self.assertEqual(output['result']['drafts'], [
            {key: SERVICE[key] for key in ('mcId', 'title', 'text')}])
        self.assertEqual(json.loads(calls[0]['body']['messages'][1]['content']),
                         {'description': description, 'mcId': 101, 'forced_categories': [103]})
        for services in ([], [dict(SERVICE, mcId=105)], [SERVICE, dict(SERVICE, mcId=105)]):
            with self.subTest(services=services), fake_model(services=services) as (endpoint, _):
                with self.assertRaises(llm.UnsupportedCategories):
                    llm.generate(description, 101, [103], endpoint=endpoint)
        with fake_model() as (endpoint, _):
            with self.assertRaises(llm.ModelError):
                llm.generate('Новый текст без прежних цитат.', 101, [103], endpoint=endpoint)

    def test_generate_empty_without_call_and_invalid_selection(self):
        with fake_model() as (endpoint, calls):
            output = llm.generate(SOURCE, 101, [], endpoint=endpoint)
        self.assertEqual(output['result'], {'drafts': []})
        self.assertEqual(output['meta']['attempts'], 0)
        self.assertEqual(calls, [])
        for selected in (None, '103', [101], [True], [103, 103], [999]):
            with self.subTest(selected=selected), self.assertRaises(llm.InputError):
                llm.generate(SOURCE, 101, selected)

    def test_cut_http_response_is_explicit_transport_error(self):
        for operation, args in ((llm.detect, (SOURCE, 101)),
                                (llm.generate, (SOURCE, 101, [103]))):
            with self.subTest(operation=operation.__name__), fake_model(truncate_response=True) as (endpoint, _):
                with self.assertRaises(llm.TransportError):
                    operation(*args, endpoint=endpoint)


class DetectionWebTests(unittest.TestCase):
    app = fixtures.WebTests.app
    post = fixtures.WebTests.post

    def test_http_detect_generate_and_busy(self):
        payload = {'description': SOURCE, 'mcId': 101}
        with detection_model() as (endpoint, calls), self.app(endpoint) as (url, server):
            status, output = self.post(url + '/detect', payload)
            self.assertEqual(status, 200)
            self.assertEqual(output['categories'], [CATEGORY])
            server.inference_lock.acquire()
            try:
                for path, body in (('/detect', payload),
                                   ('/generate', dict(payload, selected_mcIds=[103]))):
                    status, error = self.post(url + path, body)
                    self.assertEqual((status, error['error']['code']), (503, 'busy'))
                status, empty = self.post(url + '/generate', dict(payload, selected_mcIds=[]))
                self.assertEqual((status, empty['result']), (200, {'drafts': []}))
            finally:
                server.inference_lock.release()
            self.assertEqual(len(calls), 1)
        with fake_model() as (endpoint, calls), self.app(endpoint) as (url, _):
            status, output = self.post(url + '/generate', dict(payload, selected_mcIds=[103]))
            self.assertEqual((status, output['result']['drafts'][0]['mcId']), (200, 103))
            for body in (payload, dict(payload, selected_mcIds=None),
                         dict(payload, selected_mcIds=[101]),
                         dict(payload, selected_mcIds=[103], extra=True)):
                status, error = self.post(url + '/generate', body)
                self.assertEqual((status, error['error']['code']), (422, 'invalid_input'))
            self.assertEqual(len(calls), 1)

    def test_http_model_error_is_explicit(self):
        with fake_model(services=[]) as (endpoint, _), self.app(endpoint) as (url, _):
            status, output = self.post(url + '/generate',
                                      {'description': SOURCE, 'mcId': 101, 'selected_mcIds': [103]})
            self.assertEqual((status, output['error']['code']), (502, 'unsupported_categories'))
        with fake_model(content='broken') as (endpoint, _), self.app(endpoint) as (url, _):
            status, output = self.post(url + '/detect', {'description': SOURCE, 'mcId': 101})
            self.assertEqual((status, output['error']['code']), (502, 'model_error'))


if __name__ == '__main__':
    unittest.main()
