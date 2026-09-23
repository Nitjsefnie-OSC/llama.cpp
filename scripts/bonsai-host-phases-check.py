#!/usr/bin/env python3
"""Validate bounded host-phase logs. Structural validity is not capture completeness."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

CONTRACT = {
    'schema_document_version': 2, 'wire_version': 1,
    'schema_sha256': '717e2d087e52b28acd27b1cb23ba6d739f5937d6a9a32295f1915a90f262eb98',
    'source_ready_sha256': '6bed801ac3a62fcb72a32619d4f1099c0bb5f5e6320b9bc6dc182375643d49e7',
    'source_patch_sha256': '21b4b96a4ddf2e83e4413accf696d4ba138559681e37c216a0ff180ce1dfba94',
    'source_review_sha256': '2939772485fb2d8f939a8be188923de29365887e79a42ed8e4b4695511108823',
}
MARKER = re.compile(r'(?<![A-Za-z0-9_])CUDA_HOST_PHASES\b')
COMMON = set('v event pid tid thread batch'.split())
FIELDS = {
    'batch': set('clock unit records dropped budget used valid flush_start_us'.split()),
    'span': set('id parent phase reason start_us end_us ctx lifetime attempt ubatch tokens outputs graph uid key backend device mode invocation bytes direction stream result_known result unwind'.split()),
    'aggregate': set('phase reason direction count bytes sum_us max_us'.split()),
    'end': set('records aggregates dropped valid flush_end_us'.split()),
}
POINTERS = set('ctx graph key backend'.split())
SIGNED = set('tokens outputs device result start_us end_us flush_start_us flush_end_us sum_us max_us'.split())
STRINGS = set('event clock unit phase reason mode direction stream'.split()) | POINTERS
CONTEXT = tuple('ctx lifetime attempt ubatch tokens outputs'.split())
GRAPH = tuple('graph uid key backend device mode invocation'.split())
BINDING = CONTEXT + GRAPH
GETTERS = {
    'llama_synchronize', 'llama_get_logits', 'llama_get_logits_ith',
    'llama_get_embeddings', 'llama_get_embeddings_ith', 'llama_get_embeddings_seq',
    'llama_get_embeddings_nextn', 'llama_get_embeddings_nextn_ith', 'llama_get_embeddings_layer_inp',
    'llama_get_embeddings_capture', 'llama_get_embeddings_capture_ith',
    'llama_get_sampled_token_ith', 'llama_get_sampled_probs_ith', 'llama_get_sampled_logits_ith',
    'llama_get_sampled_candidates_ith', 'llama_get_sampled_candidates_count_ith',
    'llama_get_sampled_logits_count_ith', 'llama_get_sampled_probs_count_ith',
}
BUFFER_REASONS = {'ggml_backend_cuda_buffer_'+s for s in
                  ('set_tensor', 'get_tensor', 'set_tensor_2d', 'get_tensor_2d')}
ASYNC_REASONS = {'ggml_backend_cuda_'+s for s in
                 ('set_tensor_async', 'get_tensor_async', 'set_tensor_2d_async', 'get_tensor_2d_async')}
REASONS = {
    'decode': {'llama_context_decode'}, 'ubatch': {'decode_ubatch'},
    'input_setup': {'graph_set_inputs'}, 'input_one': {'set_input'},
    'tensor_upload': {'ggml_backend_tensor_set'},
    'tensor_upload_async': {'ggml_backend_tensor_set_async'},
    'scheduler_enqueue': {'ggml_backend_sched_graph_compute_async'},
    'scheduler_sync': {'ggml_backend_sched_synchronize'}, 'logits_enqueue': {'logits_d2h'},
    'context_sync': {'synchronize'}, 'output_getter': GETTERS,
    'cuda_buffer_copy': BUFFER_REASONS, 'cuda_buffer_sync': BUFFER_REASONS,
    'cuda_async_copy': ASYNC_REASONS, 'cuda_backend_sync': {'ggml_backend_cuda_synchronize'},
    'cuda_graph_compute': {'ggml_backend_cuda_graph_compute'},
    'cuda_graph_submit': {'selected_graph_mode'}, 'cuda_graph_launch': {'cudaGraphLaunch'},
}
CUDA_IO = {'cuda_buffer_copy', 'cuda_buffer_sync', 'cuda_async_copy', 'cuda_backend_sync'}
# This is an implementation-V1 fact, not a universal schema restriction. The
# calls before mode selection inspect/cache/log; none contains a tracing hook.
# A new early hook requires a newly bound table, preserving its unbound identity.
EARLY_CUDA_CHILDREN = frozenset()
LIMITS = [
    'Only structural consistency is validated. Actual source/binary provenance and expected run coverage require independent receipts.',
    'A wholly missing final buffered root after process abort cannot be detected; parser success is not capture completeness.',
    'Intervals are host wall time, not CPU-active time, GPU duration, or GPU idle time; no CUPTI epoch mapping exists.',
    'Inclusive parent/child durations and API bytes overlap; do not sum them. Per-phase unions can also overlap.',
    'Exclusive sums are additive within a synchronous thread tree, but concurrent threads can overlap in wall time.',
    'flush_end_us precedes its own end-record logging. That write and helper return cost remain unattributed perturbation.',
    'Logger intervals measure producer formatting/callback/enqueue/backpressure, not asynchronous file-sink completion or worker logging cost.',
    'Sampler/server work outside measured scopes is uninstrumented. Enabled tracing and logging perturb execution.',
    'Early-child attribution is bound to source V1, whose permitted early-child table is empty.',
    'Decode accounting uses source V1 batch indices, not graph reservation or KV padding. Failed/unwound final ubatch completion is not established.',
    'Successful decode with outputs=-1 and no decode ubatches identifies the source V1 encode fallback; its encoder work has no ubatch scopes.',
]


def require(condition, message):
    if not condition: raise ValueError(message)


def number(value, signed=False):
    pattern = r'(?:0|[1-9][0-9]*|-[1-9][0-9]*)' if signed else r'(?:0|[1-9][0-9]*)'
    require(re.fullmatch(pattern, value) is not None, 'noncanonical integer: '+value)
    n = int(value)
    require(-(1 << 63) <= n < (1 << (63 if signed else 64)), 'integer out of range')
    return n


def parse(line):
    matches = list(MARKER.finditer(line))
    if not matches: return None
    require(len(matches) == 1, 'multiple markers on one line')
    text = line[matches[0].end():].rstrip('\r\n')
    require(text.startswith(','), 'missing marker delimiter')
    raw = {}
    for part in text[1:].split(','):
        k, sep, v = part.partition('=')
        require(sep and k and v and k not in raw, 'missing/duplicate field')
        raw[k] = v
    require(raw.get('event') not in ('allocation_failure', 'exhausted'), 'fatal recording failure: '+raw.get('event', 'missing'))
    event = raw.get('event')
    require(event in FIELDS and set(raw) == COMMON | FIELDS[event], 'unknown event or missing/extra fields')
    r = {k:(v if k in STRINGS else number(v, k in SIGNED)) for k,v in raw.items()}
    for key in POINTERS & r.keys():
        require(re.fullmatch(r'0x(?:0|[1-9a-f][0-9a-f]*)', r[key]) is not None, 'noncanonical pointer')
        require(int(r[key],16) < 1 << 64, 'pointer out of range')
    require(r['v'] == 1 and 0 < r['pid'] < 1 << 32 and 0 < r['tid'] < 1 << 32, 'wire version/native PID/TID')
    require(r['thread'] > 0 and r['batch'] > 0, 'unbound thread/batch')
    for key in {'valid','result_known','unwind'} & r.keys(): require(r[key] in (0,1), 'invalid Boolean '+key)
    for key in SIGNED & r.keys():
        require(r[key] >= (-1 if key in ('tokens','outputs','device') else -(1 << 63) if key=='result' else 0), 'invalid signed field '+key)
    if 'phase' in r:
        require(r['phase'] in REASONS and r['reason'] in REASONS[r['phase']], 'unknown phase/reason')
        direction = 'none'
        if r['phase'] == 'logits_enqueue': direction = 'd2h'
        if r['phase'] in ('cuda_buffer_copy','cuda_async_copy'):
            direction = 'h2d' if '_set_' in r['reason'] else 'd2h'
        require(r['direction'] == direction, 'phase/direction contradiction')
    if event == 'span':
        require(r['id'] > 0 and r['end_us'] >= r['start_us'], 'span identity/time')
        require(r['mode'] in ('unbound','direct','capture','replay'), 'unknown mode')
        stream = 'per_thread' if r['phase'] in ('cuda_buffer_copy','cuda_buffer_sync') else 'backend_existing' if r['phase'] in ('cuda_async_copy','cuda_backend_sync') else 'unknown'
        require(r['stream'] == stream, 'phase/stream contradiction')
        if r['phase'] not in ('tensor_upload','tensor_upload_async','logits_enqueue','cuda_buffer_copy','cuda_async_copy'):
            require(r['bytes'] == 0, 'non-transfer bytes')
        if r['phase'] == 'decode':
            require(r['result_known'] or r['unwind'], 'decode lacks result/unwind')
            if r['result_known']: require(r['result'] in (-3,-2,-1,0,1,2), 'unexpected decode result')
        else: require(r['result_known'] == 0 and r['result'] == 0, 'unexpected phase result')
        if not r['result_known']: require(r['result'] == 0, 'unknown result must be zero')
        if r['ctx'] == '0x0':
            require(tuple(r[k] for k in CONTEXT[1:]) == (0,0,0,-1,-1), 'unbound context contradiction')
        else: require(r['lifetime'] > 0, 'bound context lacks lifetime')
        if r['graph'] == '0x0': require(r['uid'] == 0 and r['key'] == '0x0', 'null graph has UID/key')
        if r['mode'] == 'unbound': require(r['invocation'] == 0, 'unbound mode has invocation')
        else:
            require(r['graph'] != '0x0' and r['backend'] != '0x0' and r['device'] >= 0 and r['invocation'] > 0, 'incomplete graph binding')
            if r['mode'] != 'direct': require(r['key'] != '0x0', 'captured graph lacks key')
        require((r['backend']=='0x0') == (r['device']==-1), 'backend/device contradiction')
        require(r['device'] < 1 << 31, 'device out of range')
    return r


def union(intervals):
    total = 0; end = None
    for a,b in sorted(intervals):
        total += max(0, b-max(a,end if end is not None else a))
        end = max(b,end if end is not None else b)
    return total


def same(a, b, keys): return all(a[k] == b[k] for k in keys)


def inheritance(r, parent):
    phase = r['phase']
    if parent and parent['phase']=='cuda_graph_compute' and phase!='cuda_graph_submit':
        require(phase in EARLY_CUDA_CHILDREN, 'no early-child hook in bound source V1')
    if phase == 'decode':
        require(parent is None and r['ctx']!='0x0' and r['attempt']>0 and r['ubatch']==0, 'decode binding/root')
    elif phase == 'ubatch':
        require(parent and parent['phase']=='decode' and same(r,parent,CONTEXT[:3]), 'ubatch context/attempt')
        require(r['ubatch']>0 and r['tokens']>0 and
                (-1 if r['unwind'] else 0)<=r['outputs']<=r['tokens'], 'ubatch dimensions')
    elif parent:
        require(same(r,parent,CONTEXT), 'foreign inherited context/attempt/ubatch')
    elif phase in ('context_sync','output_getter'):
        require(r['ctx']!='0x0' and r['ubatch']==0 and r['tokens']==-1, 'root readiness binding')
    else: require(r['ctx']=='0x0', 'unexpected root context origin')

    if phase == 'scheduler_enqueue':
        require(r['graph']!='0x0' and r['key']=='0x0' and r['backend']=='0x0' and r['mode']=='unbound', 'full-graph bridge')
    elif phase == 'cuda_graph_compute':
        # An exception before selection leaves the parent graph binding intact.
        if r['mode']=='unbound':
            require(r['unwind'] and (same(r,parent,GRAPH) if parent else r['graph']=='0x0'), 'compute lacks actual mode')
        else: require(r['invocation']>0, 'compute invocation')
    elif parent:
        keys = tuple(k for k in GRAPH if k not in ('backend','device')) if phase in CUDA_IO else GRAPH
        require(same(r,parent,keys), 'foreign inherited graph/key/mode/invocation; no V1 early-child hook')
    else:
        require(r['graph']=='0x0' and r['mode']=='unbound', 'unexpected root graph origin')
        if phase not in CUDA_IO: require(r['backend']=='0x0', 'unexpected root backend origin')
    if phase in CUDA_IO: require(r['backend']!='0x0' and r['device']>=0, 'CUDA IO lacks backend')
    if phase == 'cuda_graph_submit': require(parent and parent['phase']=='cuda_graph_compute' and r['mode']!='unbound', 'submit parent/mode')
    if phase == 'cuda_graph_launch':
        require(parent and parent['phase']=='cuda_graph_submit' and r['mode'] in ('capture','replay'), 'launch parent/mode')
    if phase == 'input_one': require(parent and parent['phase']=='input_setup', 'input parent')


def transfer_children(request, children):
    """Source V1 1D API forwarding, including opaque CPU and sync fallbacks."""
    phase=request['phase']
    if phase not in ('logits_enqueue','tensor_upload','tensor_upload_async'): return
    get=phase=='logits_enqueue';op='get' if get else 'set';direction='d2h' if get else 'h2d'
    callback={
        'copy':('cuda_buffer_copy',f'ggml_backend_cuda_buffer_{op}_tensor'),
        'wait':('cuda_buffer_sync',f'ggml_backend_cuda_buffer_{op}_tensor'),
        'async':('cuda_async_copy',f'ggml_backend_cuda_{op}_tensor_async'),
        'sync':('cuda_backend_sync','ggml_backend_cuda_synchronize'),
        'upload':('tensor_upload','ggml_backend_tensor_set'),
    }
    observed=[]
    for child in children:
        matches=[k for k,v in callback.items() if v==(child['phase'],child['reason'])]
        require(len(matches)==1, 'illegal immediate transfer callback')
        kind=matches[0];observed.append(kind)
        if kind in ('copy','async','upload'):
            require(child['bytes']==request['bytes'], 'request/callback byte mismatch')
            require(child['direction']==('none' if kind=='upload' else direction), 'request/callback direction mismatch')
    # CPU callbacks are uninstrumented. Async fallback invokes synchronize then
    # the generic 1D API; the synchronization itself may have no CUDA hook.
    if phase=='tensor_upload':
        paths=[(),('copy','wait')] if request['bytes'] else [()]
    elif get:
        paths=[(),('async',),('sync',)]
        if request['bytes']: paths += [('copy','wait'),('sync','copy','wait')]
    else:
        paths=[(),('async',),('upload',),('sync','upload')]
    observed=tuple(observed)
    require(any(observed==path or (request['unwind'] and observed==path[:len(observed)])
                for path in paths), 'transfer callback order/path mismatch')
    for a,b in zip(children,children[1:]):
        if a['phase']=='cuda_buffer_copy' and b['phase']=='cuda_buffer_sync':
            require(same(a,b,('backend','device')), 'buffer copy/wait backend mismatch')


def decode_accounting(root, children):
    """Logical batch counts: allocator splits original indices without padding.

    llama-batch.cpp init/get_n_tokens/ubatch_add retain input token count; split
    methods partition those indices. reserve_graph rounds only reservation
    dimensions, and KV padding affects storage. Neither changes these scopes.
    """
    ubatches=[x for x in children if x['phase']=='ubatch']
    require([x['ubatch'] for x in ubatches]==list(range(1,len(ubatches)+1)), 'ubatch sequence')
    nt,no=root['tokens'],root['outputs'];success=root['result_known'] and root['result']==0
    require(nt>=0 and -1<=no<=nt, 'decode token/output dimensions')
    require(not (root['result_known'] and root['unwind']), 'decode returned and unwound')
    if nt==0:
        require(not ubatches and no==-1 and (root['unwind'] or root['result']==-1), 'zero-token decode result/accounting')
    if no==-1: require(not ubatches, 'decode ubatches before output accounting')
    token_sum=sum(x['tokens'] for x in ubatches)
    output_sum=sum(max(0,x['outputs']) for x in ubatches)
    require(token_sum<=nt and (no==-1 or output_sum<=no), 'attempted ubatches exceed decode counts')
    require(not any(x['unwind'] for x in ubatches[:-1]), 'continued after unwound ubatch')
    if ubatches and ubatches[-1]['unwind']: require(root['unwind'], 'ubatch unwind lacks decode unwind')
    if success:
        require(nt>0, 'successful zero-token decode')
        if no>=0:
            require(ubatches and token_sum==nt and output_sum==no, 'completed decode/ubatch count mismatch')
        kind='encode_fallback' if no==-1 else 'completed_decoder'
    else: kind='unwind' if root['unwind'] else 'failed_decode'
    # A later ubatch proves its predecessor completed. A failed last attempt
    # has no success marker; returning normally from its RAII scope is not one.
    return dict(pid=root['pid'],thread=root['thread'],id=root['id'],attempt=root['attempt'],
                kind=kind,observed_ubatches=len(ubatches),attempted_tokens=token_sum,
                known_attempted_outputs=output_sum,
                completed_ubatches=len(ubatches) if success else max(0,len(ubatches)-1),
                completed_ubatches_is_lower_bound=not success,
                last_ubatch_completion='complete' if success and ubatches else 'not_established')


class Validator:
    def __init__(self):
        self.threads = {}; self.active = {}; self.batches = []; self.timeline = []
        self.identities = {}; self.contexts = {}; self.line_count = 0
        self.decode_accounts = []

    def claim(self, pid, ident, owner):
        key = (pid,ident)
        require(key not in self.identities or self.identities[key]==owner, 'reused process identity')
        self.identities[key] = owner

    def feed(self, r):
        key = (r['pid'],r['thread'])
        if r['event']=='batch':
            require(key not in self.active, 'overlapping batch frame')
            previous = self.threads.get(key, dict(batch=0,used=0,tid=r['tid'],end=0))
            require(previous['tid']==r['tid'] and r['batch']==previous['batch']+1, 'thread identity/batch sequence')
            require(r['clock']=='ggml_time_us' and r['unit']=='us', 'clock/unit')
            require(r['budget']==262144 and 0<r['records']<=4096, 'budget/capacity')
            require(r['used']==previous['used']+r['records'] and r['used']<=r['budget'], 'cumulative used mismatch')
            require(r['dropped']==0 and r['valid']==1, 'dropped/invalid batch')
            self.claim(r['pid'],r['thread'],('thread',r['tid']))
            self.active[key] = dict(header=r,previous=previous,spans=[],aggregates={},stage='spans')
            return
        require(key in self.active, 'record outside batch frame')
        b = self.active[key]; h = b['header']
        require(same(r,h,('pid','tid','thread','batch')), 'foreign batch identity')
        if r['event']=='span':
            require(b['stage']=='spans' and len(b['spans'])<h['records'], 'extra/late span')
            require(r['id']==b['previous']['used']+len(b['spans'])+1, 'noncontiguous/duplicate span ID')
            b['spans'].append(r)
        elif r['event']=='aggregate':
            require(len(b['spans'])==h['records'], 'aggregate before all spans')
            b['stage']='aggregates';k=(r['phase'],r['reason'],r['direction'])
            require(k not in b['aggregates'], 'duplicate aggregate');b['aggregates'][k]=r
        else:
            require(r['records']==h['records'] and r['aggregates']==len(b['aggregates']), 'footer count mismatch')
            require(r['dropped']==0 and r['valid']==1 and r['flush_end_us']>=h['flush_start_us'], 'invalid footer/clock')
            self.close(b,r);del self.active[key]
            self.threads[key]=dict(batch=r['batch'],used=h['used'],tid=r['tid'],end=r['flush_end_us'])

    def close(self, b, footer):
        h=b['header'];ss=b['spans'];require(len(ss)==h['records'], 'missing spans')
        require(ss[0]['parent']==0, 'missing root')
        require(ss[0]['start_us']>=b['previous']['end'] and h['flush_start_us']>=ss[0]['end_us'], 'nonmonotonic batch/flush clock')
        stack=[];children=defaultdict(list);expected=defaultdict(lambda:dict(count=0,bytes=0,sum_us=0,max_us=0))
        for i,r in enumerate(ss):
            require(i==0 or r['parent']>0, 'multiple roots')
            while stack and stack[-1]['id']!=r['parent']:
                require(stack.pop()['end_us']<=r['start_us'], 'crossing/reopened synchronous scope')
            parent=stack[-1] if stack else None
            require((parent['id'] if parent else 0)==r['parent'], 'missing/closed parent')
            if parent:
                require(parent['start_us']<=r['start_us']<=r['end_us']<=parent['end_us'], 'child outside parent')
                children[parent['id']].append(r)
            inheritance(r,parent);stack.append(r)
            if r['ctx']!='0x0':
                self.claim(r['pid'],r['lifetime'],('context',r['ctx']))
                self.contexts.setdefault((r['pid'],r['lifetime']),[]).append(r)
            if r['phase']=='cuda_graph_compute' and r['invocation']:
                self.claim(r['pid'],r['invocation'],('invocation',r['thread'],r['id']))
            g=expected[(r['phase'],r['reason'],r['direction'])];duration=r['end_us']-r['start_us']
            g['count']+=1;g['bytes']+=r['bytes'];g['sum_us']+=duration;g['max_us']=max(g['max_us'],duration)
        require(set(expected)==set(b['aggregates']), 'missing/extra aggregate keys')
        for k,g in expected.items(): require(all(b['aggregates'][k][f]==v for f,v in g.items()), 'aggregate count/bytes/sum/max mismatch')
        for r in ss:
            direct=children[r['id']]
            transfer_children(r,direct)
            if r['phase']=='decode':
                self.decode_accounts.append(decode_accounting(r,direct))
            if r['phase']=='cuda_graph_compute' and not r['unwind']:
                require(sum(x['phase']=='cuda_graph_submit' for x in direct)==1, 'missing/extra submit')
            if r['phase']=='cuda_graph_submit' and not r['unwind']:
                require(sum(x['phase']=='cuda_graph_launch' for x in direct)==(r['mode']!='direct'), 'launch coverage/mode')
            r=dict(r,exclusive_us=(r['end_us']-r['start_us'])-union((x['start_us'],x['end_us']) for x in direct))
            self.timeline.append(r)
        self.batches.append(dict(pid=h['pid'],tid=h['tid'],thread=h['thread'],batch=h['batch'],
                                 root_start_us=ss[0]['start_us'],root_end_us=ss[0]['end_us'],
                                 flush_start_us=h['flush_start_us'],flush_end_us=footer['flush_end_us']))

    def finish(self):
        require(not self.active, 'truncated batch(s)')
        require(self.batches, 'no host-phase batches')
        for records in self.contexts.values():
            decodes=sorted((r for r in records if r['phase']=='decode'),key=lambda r:r['start_us'])
            require([r['attempt'] for r in decodes]==list(range(1,len(decodes)+1)), 'context decode attempts missing/reused')
            for a,b in zip(decodes,decodes[1:]): require(a['end_us']<=b['start_us'], 'overlapping context decodes')
            for r in records:
                if r['phase'] not in ('context_sync','output_getter') or r['parent']!=0: continue
                last=[d for d in decodes if d['start_us']<=r['start_us']]
                require(r['attempt']==(last[-1]['attempt'] if last else 0), 'foreign getter/readiness attempt')
        pointers=defaultdict(list)
        for (pid,lifetime),records in self.contexts.items():
            pointers[(pid,records[0]['ctx'])].append((min(r['start_us'] for r in records),max(r['end_us'] for r in records)))
        for lifetimes in pointers.values():
            lifetimes.sort()
            require(all(a[1]<=b[0] for a,b in zip(lifetimes,lifetimes[1:])), 'overlapping reused context pointer lifetimes')
        # Native TID reuse is legal only after the old thread's observed lifetime.
        native=defaultdict(lambda:defaultdict(list))
        for b in self.batches: native[(b['pid'],b['tid'])][b['thread']].append(b)
        for lifetimes in native.values():
            intervals=sorted((min(b['root_start_us'] for b in bs),max(b['flush_end_us'] for b in bs)) for bs in lifetimes.values())
            require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'overlapping native thread lifetimes')
        phases={}
        for phase in sorted({r['phase'] for r in self.timeline}):
            rows=[r for r in self.timeline if r['phase']==phase]
            phases[phase]=dict(count=len(rows),inclusive_sum_us=sum(r['end_us']-r['start_us'] for r in rows),
                               exclusive_sum_us=sum(r['exclusive_us'] for r in rows),api_bytes=sum(r['bytes'] for r in rows))
        processes={}
        for pid in sorted({b['pid'] for b in self.batches}):
            bs=[b for b in self.batches if b['pid']==pid]
            processes[str(pid)]=dict(covered_wall_union_us=union((b['root_start_us'],b['root_end_us']) for b in bs),
                                    logger_wall_union_us=union((b['flush_start_us'],b['flush_end_us']) for b in bs))
        return dict(structurally_valid=True,external_coverage={'complete':False,'status':'not_established',
                    'requires':['successful external run receipts','independent expected decode/graph coverage','actual source/binary provenance']},
                    source_contract=CONTRACT,batches=len(self.batches),spans=len(self.timeline),per_process=processes,
                    phases=phases,graph_modes=dict(Counter(r['mode'] for r in self.timeline if r['phase']=='cuda_graph_compute')),
                    decode_results=dict(Counter(str(r['result']) if r['result_known'] else 'unwind' for r in self.timeline if r['phase']=='decode')),
                    decode_accounting=self.decode_accounts,
                    unbound_context_spans=sum(r['ctx']=='0x0' for r in self.timeline),
                    timeline=self.timeline,logger_intervals=self.batches,limitations=LIMITS)


def analyze(lines):
    validator=Validator()
    for i,line in enumerate(lines,1):
        try:
            r=parse(line)
            if r is not None: validator.feed(r)
        except ValueError as error: raise ValueError(f'line {i}: {error}') from error
    return validator.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        with args.output.open('x',encoding='utf-8') as output:
            result={'structurally_valid':False,'external_coverage':{'complete':False,'status':'not_established'},
                    'source_contract':CONTRACT,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'input_log':str(args.log.resolve()),'limitations':LIMITS}
            try:
                data=args.log.read_bytes();result['input_sha256']=hashlib.sha256(data).hexdigest()
                result.update(analyze(data.decode('utf-8-sig').splitlines(keepends=True)))
                code=0
            except (ValueError,OSError,UnicodeError) as error:
                result['error']=str(error);code=2
            json.dump(result,output,indent=2);output.write('\n')
        print(f"structurally_valid={result['structurally_valid']}; external_coverage=not_established; output={args.output}")
        if code: print(result['error'],file=sys.stderr)
        return code
    except OSError as error:
        print(str(error),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
