"""Resolve native numpy field casing from its actual dtype, no physics changes."""
from pathlib import Path

previous = Path(__file__).with_name('contact_point_fixture_v15_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())
old = '        states = []; forces = []; readings = []; contact_records = []; contact_errors = []\n'
assert source.count(old) == 1
source = source.replace(old, old + '''        contact_dtypes = []
        def contact_field(record, logical):
            normalized = logical.replace('_','').lower()
            matches = [name for name in record.dtype.names if name.replace('_','').lower()==normalized]
            assert len(matches)==1, (logical,record.dtype.names)
            return record[matches[0]]
''')
old = '                    names = contacts.dtype.names\n'
assert source.count(old) == 1
source = source.replace(old, old + '                    if contacts.dtype.names not in contact_dtypes: contact_dtypes.append(contacts.dtype.names)\n')
for name in ('body0','body1','env0','env1','normal','lambda','local_pos0','local_pos1','offset0','offset1'):
    old = "contact_record['" + name + "']"
    assert source.count(old) == 1
    source = source.replace(old, "contact_field(contact_record,'" + name + "')")
source = source.replace('native = dict(status=native_status, records=contact_records, errors=contact_errors,',
                        'native = dict(status=native_status, records=contact_records, errors=contact_errors, dtype_fields=contact_dtypes,')
exec(compile(source, str(Path(__file__)), 'exec'), globals())
