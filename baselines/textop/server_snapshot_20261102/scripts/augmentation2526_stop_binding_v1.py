"""Fixed provenance for the user-interrupted augmentation run."""

EXPECTED = {
    'request': '419011b4909eea18dfe8eb7a698b8591d0b52d7261d5612875eea941c69ec806',
    'receipt': '6617cdecaf48dc97f9867c2307d1601b34b7183d761b1a07916fd3426be4e933',
    'exit': '016474760e7d9437ef26ad372114ffd9c237010cc0a3a5887bd8883677a482ed',
    'process': 'a60c692968e2f3e7689d1f12ab36b791ddb72c32b360c10d2c058c5d56b3e58f',
    'outer': '857a96c9474624cec4418bb86e755613a5fb3430bf5b3478354415d1030ec7b2',
    'resources': 'f7f99b4b8655bd03be61c8500afdbece83b49225956f115ef7f2157ea3a906e7',
}


def validate_interrupted_binding(observed):
    if not isinstance(observed, dict) or observed != EXPECTED:
        raise ValueError('interrupted producer closure does not match observed evidence')
