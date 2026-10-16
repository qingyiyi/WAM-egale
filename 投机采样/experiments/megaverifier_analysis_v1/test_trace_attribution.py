"""CPU-only synthetic accounting tests; no model or GPU execution."""
import copy
import math
import unittest
from trace_attribution import analyze_document


def cpu(name,ts,dur,ext=None,corr=None,cat='cpu_op',pid=10,tid=10):
    args={}
    if ext is not None:args['External id']=ext
    if corr is not None:args['correlation']=corr
    return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=pid,tid=tid,args=args)


def kernel(ts=120,dur=5,ext=4,corr=7,stream=7):
    return dict(ph='X',cat='kernel',name='physical_kernel',ts=ts,dur=dur,pid=0,tid=stream,
                args={'External id':ext,'correlation':corr,'device':0,'context':1,'stream':stream})


def scopes():
    return [cpu('mv/candidate/short/00',0,100,1,cat='user_annotation'),
            cpu('mv/block/00/block',10,80,2,cat='user_annotation'),
            cpu('mv/block/00/attn1',20,30,3,cat='user_annotation')]


def example():
    return scopes()+[cpu('aten::linear',21,28,9),cpu('aten::mm',23,20,4),
                     cpu('cudaLaunchKernel',25,3,4,7,cat='cuda_runtime'),kernel(),
                     dict(ph='s',cat='ac2g',id=7,name='flow',ts=26,pid=10,tid=10),
                     dict(ph='f',cat='ac2g',id=7,name='flow',ts=120,pid=0,tid=7)]


class AttributionTests(unittest.TestCase):
    def test_nested_operator_totals_not_added(self):
        r=analyze_document(example())
        self.assertEqual(r['unique_kernel_count'],1)
        self.assertEqual(r['unique_kernel_duration_sum_us'],5)
        self.assertEqual(r['kernels'][0]['scope'],'block00/attn1')
        self.assertEqual(r['kernels'][0]['operator'],'aten::mm')
        self.assertEqual(r['kernels'][0]['link_methods'],['correlation','external_id','flow'])

    def test_cpu_gpu_projection_not_duplicate_call(self):
        es=example()
        for e in scopes():
            g=copy.deepcopy(e);g.update(cat='gpu_user_annotation',pid=0,tid=7,ts=110,dur=30);es.append(g)
        r=analyze_document(es)
        self.assertEqual(r['unassigned_kernel_count'],0)
        self.assertEqual(r['kernels'][0]['scope'],'block00/attn1')
        self.assertEqual(r['marker_categories']['user_annotation']['mv/block/00/block'],1)
        self.assertEqual(r['marker_categories']['gpu_user_annotation']['mv/block/00/block'],1)

    def test_only_exact_activity_duplicates_removed(self):
        es=example();es.extend([copy.deepcopy(kernel()),kernel(ts=140),kernel(stream=8)])
        r=analyze_document(es)
        self.assertEqual(r['raw_kernel_events'],4)
        self.assertEqual(r['unique_kernel_count'],3)
        self.assertEqual(r['duplicate_activity_count'],1)
        self.assertEqual(r['unique_kernel_duration_sum_us'],15)

    def test_outside_modules_pointwise(self):
        r=analyze_document(scopes()+[cpu('aten::add',60,5,4),kernel(corr=None)])
        self.assertEqual(r['kernels'][0]['bucket'],'block00/outside_modules/pointwise')

    def test_external_parent_does_not_inherit_midpoint_child(self):
        es=scopes()+[cpu('aten::outer',11,78,4),kernel(corr=None)]
        r=analyze_document(es)
        self.assertEqual(r['kernels'][0]['scope'],'block00/outside_modules')
        self.assertEqual(r['kernels'][0]['operator'],'aten::outer')

    def test_external_candidate_stays_candidate(self):
        r=analyze_document(scopes()+[kernel(ext=1,corr=None)])
        self.assertEqual(r['kernels'][0]['scope'],'candidate/outside_blocks')

    def test_conflicting_links_fail_closed(self):
        es=example()+[cpu('mv/block/01/block',101,50,12,cat='user_annotation'),
                     cpu('mv/candidate/short/00',100,60,13,cat='user_annotation'),
                     cpu('aten::mm',110,20,4)]
        r=analyze_document(es)
        self.assertEqual(r['unassigned_kernel_count'],1)
        self.assertEqual(r['kernels'][0]['unassigned_reason'],'conflicting_cpu_scope_links')

    def test_missing_links_unassigned(self):
        r=analyze_document(scopes()+[kernel(ext=999,corr=999)])
        self.assertEqual(r['unassigned_kernel_count'],1)
        self.assertEqual(r['unassigned_duration_fraction'],1)

    def test_same_scope_conflicting_operator_unknown(self):
        r=analyze_document(scopes()+[cpu('aten::add',55,5,4),cpu('aten::mul',65,5,4),kernel(corr=None)])
        self.assertEqual(r['kernels'][0]['scope'],'block00/outside_modules')
        self.assertEqual(r['kernels'][0]['operator_group'],'unknown_operator')

    def test_stream_overlap_union_and_big_timestamps(self):
        r=analyze_document([kernel(ts=1000000000000.1,dur=.003),kernel(ts=1000000000000.101,dur=.003,stream=8)])
        self.assertEqual(r['unique_kernel_duration_sum_us'],.006)
        self.assertEqual(r['device_union_kernel_busy_us'],.004)

    def test_non_kernel_gpu_copy_excluded(self):
        es=example();c=kernel();c.update(cat='gpu_memcpy',name='Memcpy HtoD');es.append(c)
        r=analyze_document(es)
        self.assertEqual(r['unique_kernel_count'],1)
        self.assertEqual(r['cuda_copy_set_events_excluded'],1)

    def test_ambiguous_flow_not_used(self):
        es=example();es=[x for x in es if x.get('cat') not in {'cpu_op','cuda_runtime'}]
        es.append(dict(ph='s',cat='ac2g',id=7,name='flow',ts=27,pid=10,tid=10))
        r=analyze_document(es)
        self.assertEqual(r['unassigned_kernel_count'],1)
        self.assertEqual(r['flow_diagnostics']['nonunique_or_missing_start_groups'],1)

    def test_nonfinite_duration_rejected(self):
        for bad in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):analyze_document([kernel(dur=bad)])

    def test_no_kernels_rejected(self):
        with self.assertRaises(ValueError):analyze_document(scopes())


if __name__=='__main__':unittest.main(verbosity=2)
