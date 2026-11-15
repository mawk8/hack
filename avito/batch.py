"""Bounded CSV import; titles are display metadata, never model input."""
import csv
import io

MAX_CSV_BYTES = 2 * 1024 * 1024


def parse_csv(raw):
    if not 0 < len(raw) <= MAX_CSV_BYTES:
        raise ValueError('Нужен непустой CSV размером до 2 MiB.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError as error:
        raise ValueError('CSV должен быть в UTF-8 (допустим BOM).') from error
    # Only the header chooses the delimiter; punctuation in descriptions cannot change it.
    reader = None
    for delimiter in (',', ';', '\t'):
        candidate = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter, strict=True)
        try:
            header = [name.strip() for name in next(candidate)]
        except (StopIteration, csv.Error):
            continue
        if ('description' in header and len(header) == len(set(header))
                and set(header) <= {'id', 'testcase', 'title', 'description'}
                and not {'id', 'testcase'} <= set(header)):
            header = ['id' if name == 'testcase' else name for name in header]
            reader = candidate
            break
    if reader is None:
        raise ValueError('Нужны уникальные колонки description и необязательные id (или testcase), title; id и testcase вместе запрещены. Разделитель: , ; или табуляция.')
    items, seen = [], set()
    try:
        for row in reader:
            if not row:  # Blank physical lines do not count as records.
                continue
            number = len(items) + 1
            if number > 200:
                raise ValueError('Допустимо не более 200 записей.')
            if len(row) != len(header):
                raise ValueError(f'Запись {number}: число колонок не совпадает с заголовком.')
            fields = dict(zip(header, row))
            description = fields['description']
            if not description.strip() or len(description) > 12000:
                raise ValueError(f'Запись {number}: description должен содержать от 1 до 12000 символов.')
            identifier = fields['id'].strip() if 'id' in fields else str(number)
            if not identifier or identifier in seen:
                raise ValueError(f'Запись {number}: id пустой или повторяется.')
            seen.add(identifier)
            items.append({'id': identifier, 'title': fields.get('title', ''), 'description': description})
    except csv.Error as error:
        raise ValueError('Неверный CSV: проверьте кавычки и колонки.') from error
    if not items:
        raise ValueError('CSV не содержит записей.')
    return items
