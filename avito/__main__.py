"""python -m avito serve | predict."""
import argparse
import json
import sys

from .llm import AnalysisError, DEFAULT_ENDPOINT, DEFAULT_MODEL, analyze


def main():
    parser = argparse.ArgumentParser(description='Локальные черновики дополнительных услуг ремонта')
    commands = parser.add_subparsers(dest='command', required=True)
    serve_parser = commands.add_parser('serve', help='Локальное веб-демо')
    serve_parser.add_argument('--port', type=int, default=7860)
    predict_parser = commands.add_parser('predict', help='Один запрос к локальной модели')
    predict_parser.add_argument('--description', required=True)
    predict_parser.add_argument('--mcId', type=int, required=True)
    predict_parser.add_argument('--diagnostics', action='store_true')
    for command in (serve_parser, predict_parser):
        command.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
        command.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        if args.command == 'serve':
            from .web import serve
            serve(args.port, args.endpoint, args.model)
        else:
            output = analyze(args.description, args.mcId, endpoint=args.endpoint, model=args.model)
            print(json.dumps(output if args.diagnostics else output['result'], ensure_ascii=False))
    except AnalysisError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error),
                                    'elapsed_seconds': error.elapsed_seconds}}, ensure_ascii=False), file=sys.stderr)
        return 1
    except OSError:
        print('Не удалось запустить локальный сервер: проверьте порт.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
