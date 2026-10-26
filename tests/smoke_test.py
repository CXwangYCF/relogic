"""Offline integration checks using a temporary OpenAI-compatible model server."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
requests = []

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append((self.path, body))
        assert self.path == '/v1/chat/completions'
        payload = {'choices': [{'message': {'content': 'Reasoning.\nanswer {The test answer}'},
                                'finish_reason': 'stop'}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run(*args, cwd=ROOT):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(result.stdout)
    return result.stdout


def main():
    manifest = json.loads((ROOT/'dataset/manifest.json').read_text())
    for row in manifest:
        path = ROOT/'dataset/relogic_data'/row['dataset']/'raw'/f"{row['split']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert len(json.loads(path.read_text(encoding='utf-8'))) == row['count']

    templates = json.loads((ROOT/'BLEC-main/tests/template_to_names_test.json').read_text(encoding='utf-8'))[0]
    spider = json.loads((ROOT/'dataset/relogic_data/spider/raw/test.json').read_text(encoding='utf-8'))
    assert all(r['query'] in templates for r in spider), 'Incomplete Spider test templates'

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            demo = tmp/'demo.json'
            demo.write_text(json.dumps({'demo': [{'question': 'DEMO_MARKER',
                'rationale': 'Example reasoning.', 'pred_ans': 'Example answer'}]}))
            for dataset in ['logic2text', 'spider']:
                task = f'relogic_{dataset}'
                for method in ['zero_shot_cot', 'auto_cot']:
                    out = tmp/f'{dataset}_{method}'
                    run('run_inference.py', '--dataset', task, '--method', method,
                        '--model', 'mock', '--base_url', f'http://127.0.0.1:{server.server_port}/v1',
                        '--reasoning_effort', 'none', '--limit_dataset_size', 2,
                        '--demo_path', demo, '--output_dir', out)
                    rows = [json.loads(line) for line in (out/f'{task}.jsonl').read_text().splitlines()]
                    assert len(rows) == 2 and all(r['prediction'] == 'The test answer' for r in rows)
                    for _, body in requests[-2:]:
                        assert body['max_tokens'] == 10240
                        assert body['reasoning_effort'] == 'none'
                        prompt = body['messages'][0]['content']
                        assert "Let's think step by step." in prompt
                        assert ('DEMO_MARKER' in prompt) == (method == 'auto_cot')
                        assert ('Mandatory mappings:' in prompt) == (dataset == 'logic2text')
                    report = out/'blec.json'
                    run('evaluate_blec.py', '--dataset', dataset, '--input', out/f'{task}.jsonl', '--output', report)
                    summary = json.loads(report.read_text())['summary']
                    assert summary['total'] == 2 and summary['missing_templates'] == 0

                custom = tmp/f'{dataset}_custom.json'
                row = ({'logic_str': 'CUSTOM_TRAIN_MARKER', 'sent': 'reference'}
                       if dataset == 'logic2text' else {'query': 'CUSTOM_TRAIN_MARKER', 'question': 'reference'})
                custom.write_text(json.dumps([row]))
                run('run_inference.py', '--dataset', task, '--method', 'zero_shot_cot',
                    '--dataset_path', custom, '--model', 'mock',
                    '--base_url', f'http://127.0.0.1:{server.server_port}/v1',
                    '--output_dir', tmp/f'{dataset}_train')
                assert 'CUSTOM_TRAIN_MARKER' in requests[-1][1]['messages'][0]['content']
                assert 'reasoning_effort' not in requests[-1][1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    run('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_logic2text_keyword_case.py', cwd=ROOT/'BLEC-main')
    print('PASS: data checksums, all 1034 Spider templates, both datasets × two inference modes,')
    print('training input override, original token budget, thinking options, and BLEC integration.')


if __name__ == '__main__':
    main()
