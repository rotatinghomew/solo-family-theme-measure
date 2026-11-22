import json
from pathlib import Path
import unittest


class RecordsTest(unittest.TestCase):
    def test_known_input_contract(self):
        source = Path(__file__).resolve().parents[1] / "data" / "records.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        sessions = {item['session'] for item in payload['summaries']}
        households = {(item['period'], item['household']) for item in payload['frame']}
        assert all((item['period'], item['household']) in households for item in payload['summaries'])
        for item in payload['reviews']:
            assert item['session'] in sessions and set(item['labels']) <= set(payload['definitions'][str(item['taxonomy'])])
        assert len(sessions) < len(payload['summaries'])
        assert set(payload['approval']['holdout_periods']) & set(payload['approval']['mature_periods']) == set()
