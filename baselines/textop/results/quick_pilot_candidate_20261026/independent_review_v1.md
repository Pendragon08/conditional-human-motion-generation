# Independent document_v1 review (preserved before revision2)

The documented read-only invocation actually exited0 with15 checks. No SDK, training signals or optimizer updates. Six candidate files matched; both quick flags and three full-v4 flags were false; original16 and live control hashes unchanged.

The independent agent found two issues of interpretation, not an observed physical failure:

1. Reset reference0 and command advancement after termination/reset mean T−1 physical calls consume targets0..T−2. This covers elapsed duration with zero-order hold, without a separate endpointT−1 comparison. Revision2 now asserts/records the actual consumed sequence and explicitly declines an endpoint claim.
2. Requiring at least one natural completion does not guarantee a failed trial was observed if all8 pass. Documentation now states observed failures are retained and does not claim both branches were actually exercised.

Version1 source bytes and actual CPU/document receipts remain preserved. Revision2 received separate31-case local/server CPU execution and a fresh16-check read-only document pass. No actual robot trial has been performed by either preflight.
