#!/usr/bin/env python3
"""Replay archived scalar geometry/assumptions; not a released deployment estimate."""
import argparse
import json
from pathlib import Path


def ceildiv(a, b):
    return (a + b - 1) // b


def kv_snapshot(geometry, length, inflight):
    safe = length - inflight
    assert safe >= 0
    buckets = geometry['page_buckets_bytes']
    derived_slots = max(max(g['normal_owners_per_bucket']) for g in geometry['groups']) + geometry['trailing_slots']
    assert derived_slots == geometry['tuple_slots']
    unit = derived_slots * sum(buckets)
    assert unit == geometry['allocator_unit_bytes']
    groups = []
    for g in geometry['groups']:
        block, window = g['logical_block_tokens'], g['window_tokens']
        start = max(0, safe - window + 1) if window else 0
        pages = ceildiv(length, block) - start // block
        component = pages * sum(n * b for n, b in zip(g['owners_per_bucket'], buckets))
        mapped = pages * unit
        assert mapped >= component
        groups.append({'id': g['id'], 'pages': pages, 'component_bytes': component, 'mapped_bytes': mapped})
    component = sum(g['component_bytes'] for g in groups)
    mapped = sum(g['mapped_bytes'] for g in groups)
    return {'component_bytes': component, 'mapped_bytes': mapped, 'hybrid_extra_bytes': mapped - component, 'groups': groups}


def calculate(data):
    GiB = data['units']['GiB_bytes']
    weight = data['weights']['base_EP16_bytes']
    assert weight == sum(data['weights']['module_bytes'].values())
    assert weight == sum(data['weights'][k] for k in ['payload_bytes', 'scale_bytes', 'extra_identified_copy_bytes'])
    activation = max(p['bytes'] for p in data['activation']['phase_core'].values())
    assert activation == data['activation']['max_core_bytes']
    persistent = sum(data['persistent'][k] for k in ['target_MTP_hidden_bytes', 'target_shared_TopK_bytes', 'V2_dspark_combined_hidden_bytes'])
    assert persistent == data['persistent']['known_sum_bytes']
    chunk = data['kv_geometry']['single_chunk_tokens']
    multiplier = data['inputs']['user_budget_assumptions']['async_KV_multiplier']
    graph_lo, graph_hi = data['inputs']['user_budget_assumptions']['graph_pool_net_extra_GiB']
    output = []
    for archived in data['results']:
        length = archived['sequence_tokens']
        single = kv_snapshot(data['kv_geometry'], length, chunk)
        double = kv_snapshot(data['kv_geometry'], length, 2 * chunk)
        budget = multiplier * single['mapped_bytes']
        # For these archived aligned cases the user x2 reserve exceeds the
        # two-chunk known page union, but this is not a universal guarantee.
        assert budget >= double['mapped_bytes']
        cases = []
        for prior_case in archived['accounted_budget_cases']:
            B = prior_case['HCCL_BUFFSIZE']
            c = data['hccl']
            communication = c['directions'] * (2*c['ordinary_domain_buffer_MiB_per_direction'] + c['DP_domain_buffer_MiB_per_direction'] + B) * data['units']['MiB_bytes']
            subtotal = weight + activation + persistent + communication + budget
            cases.append({'HCCL_BUFFSIZE': B, 'accounted_budget_bytes': [subtotal + graph_lo*GiB, subtotal + graph_hi*GiB]})
        output.append({'label': archived['label'], 'sequence_tokens': length, 'single': single, 'two_inflight': double, 'user_KV_budget_bytes': budget, 'cases': cases, 'complete_HBM_peak_bytes': None})
    return output


def check(data, output):
    for expected, actual in zip(data['results'], output):
        assert expected['sequence_tokens'] == actual['sequence_tokens']
        assert expected['single_chunk_known_mapping_bytes'] == actual['single']['mapped_bytes']
        assert expected['single_chunk_component_bytes'] == actual['single']['component_bytes']
        assert expected['single_chunk_hybrid_extra_bytes'] == actual['single']['hybrid_extra_bytes']
        assert expected['two_inflight_known_mapping_bytes'] == actual['two_inflight']['mapped_bytes']
        assert expected['user_KV_budget_bytes'] == actual['user_KV_budget_bytes']
        for before, after in zip(expected['accounted_budget_cases'], actual['cases']):
            assert before['HCCL_BUFFSIZE'] == after['HCCL_BUFFSIZE']
            assert before['accounted_budget_bytes'] == after['accounted_budget_bytes']
    assert data['kv_geometry']['request_pool_need_bytes'] is None
    assert data['kv_geometry']['allocated_pool_bytes'] is None
    assert data['complete_HBM_peak_bytes'] is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare arithmetic to archived original results')
    parser.add_argument('--json', action='store_true', help='print computed scalar results as JSON')
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name('estimates.json').read_text())
    output = calculate(data)
    if args.check:
        check(data, output)
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print('EP16/SP reference; user KV x2; HCCL=2048; graph 1-2GiB. Partial accounted budget only.')
        print('Length | Tokens | KV x2 GiB | Accounted GiB + unresolved terms')
        for row in output:
            case = next(c for c in row['cases'] if c['HCCL_BUFFSIZE'] == 2048)
            low, high = [b/2**30 for b in case['accounted_budget_bytes']]
            print(f"{row['label']:>6} | {row['sequence_tokens']:>7} | {row['user_KV_budget_bytes']/2**30:.6f} | {low:.6f}-{high:.6f}")
        if args.check:
            print('PASS: archived bytes, category sums and explicit null limits match.')
        print('2M is capacity extrapolation beyond the local 1M reference context limit.')


if __name__ == '__main__':
    main()
