"""Correct V14 contact-record/tensor name collision, preserving the failed run."""
from pathlib import Path

previous = Path(__file__).with_name('contact_point_fixture_v14_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())
assert source.count('for contact in contacts:') == 1
assert source.count('contact[') == 10
source = source.replace('for contact in contacts:', 'for contact_record in contacts:')
source = source.replace('contact[', 'contact_record[')
exec(compile(source, str(Path(__file__)), 'exec'), globals())
