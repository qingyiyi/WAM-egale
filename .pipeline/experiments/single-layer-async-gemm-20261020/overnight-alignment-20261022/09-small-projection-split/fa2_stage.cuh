// Self/cross attention共用FA2设备主循环，计算由warp3–6完成。
// 每个任务属于一个(batch,head,query tile)，B14仍是14个独立Q16实例。
// prefix按合法slot读取，private suffix只属于自己的candidate，不能跨分支注意。
#pragma once

__device__ int fa2_splits(int batch, int q, int k, int sms) {
    const int work = batch * H * ((q + 63) / 64), capacity = sms * 2, nk = (k + 127) / 128;
    if (work >= 0.8f * capacity)
        return 1;
    const int limit = min(128, min(capacity, nk));
    float best = 0.f;
    for (int n = 1; n <= limit; ++n) {
        if (n > 1 && (nk + n - 1) / n == (nk + n - 2) / (n - 1))
            continue;
        float waves = float(work * n) / capacity;
        best = fmaxf(best, waves / ceilf(waves));
    }
    for (int n = 1; n <= limit; ++n) {
        if (n > 1 && (nk + n - 1) / n == (nk + n - 2) / (n - 1))
            continue;
        float waves = float(work * n) / capacity;
        if (waves / ceilf(waves) >= 0.85f * best)
            return n;
    }
    return 1;
}

__device__ void prepare_fa2(const Params &p, Shared &s, bool cross) {
    if (threadIdx.x == 0) {
        auto &f = s.fa2;
        f = {};
        int n = cross ? p.c : (p.cache ? int(p.info[0]) + (p.candidate ? p.q : 0) : p.q);
        f.q_ptr = bp(p, cross ? CQ : SQR);
        f.o_ptr = bp(p, cross ? CA : SA);
        f.current_k = bp(p, cross ? CK : SKR);
        f.current_v = bp(p, cross ? CV : SV);
        f.prefix_length = !cross && p.cache ? int(p.info[0]) : 0;
        f.prefix_k = p.pk;
        f.prefix_v = p.pv;
        f.valid_slots = p.slots;
        f.cfg_rows = p.candidate && !cross ? p.rows : nullptr;
        f.prefix_batch = int64_t(p.pool) * D;
        f.current_batch = int64_t(cross ? p.c : p.q) * D;
        f.k_ptr = const_cast<void *>(f.prefix_length ? f.prefix_k : f.current_k);
        f.v_ptr = const_cast<void *>(f.prefix_length ? f.prefix_v : f.current_v);
        f.q_batch_stride = f.o_batch_stride = int64_t(p.q) * D;
        f.k_batch_stride = f.v_batch_stride = f.prefix_length ? f.prefix_batch : f.current_batch;
        f.q_row_stride = f.k_row_stride = f.v_row_stride = f.o_row_stride = D;
        f.q_head_stride = f.k_head_stride = f.v_head_stride = f.o_head_stride = HD;
        f.b = p.b;
        f.h = f.h_k = H;
        f.h_h_k_ratio = 1;
        f.seqlen_q = p.q;
        f.seqlen_k = n;
        f.seqlen_q_rounded = (p.q + 127) / 128 * 128;
        f.seqlen_k_rounded = (n + 127) / 128 * 128;
        f.d = f.d_rounded = HD;
        f.scale_softmax = p.attn_scale;
        f.scale_softmax_log2 = p.attn_scale_log2;
        f.p_dropout = 1.f;
        f.p_dropout_in_uint8_t = 255;
        f.rp_dropout = 1.f;
        f.scale_softmax_rp_dropout = p.attn_scale;
        f.window_size_left = f.window_size_right = -1;
        f.is_bf16 = true;
        f.is_seqlens_k_cumulative = true;
        f.num_splits = fa2_splits(p.b, p.q, n, p.sms);
        f.softmax_lse_ptr = p.fa_lse;
        f.softmax_lseaccum_ptr = p.fa_lse_partial;
        f.oaccum_ptr = p.fa_out_partial;
        if (blockIdx.x == 0) {
            int offset = cross ? 4 : 0;
            p.fa_meta[offset] = n;
            p.fa_meta[offset + 1] = f.num_splits;
            p.fa_meta[offset + 2] = p.q;
            p.fa_meta[offset + 3] = FA_SMEM;
        }
    }
    __syncthreads();
}

template <int LogSplits> __device__ void fa2_combine_tile(Shared &s, int task) {
    wam_whole_fa2::whole_combine<FaSplit, 4, LogSplits, true>(s.fa2, task,
                                                              reinterpret_cast<char *>(&s.tile));
}

__device__ void independent_attention(const Params &p, Shared &s, cg::grid_group &g, int st, bool cross) {
    prepare_fa2(p, s, cross);
    const auto &f = s.fa2;
    const bool split = f.num_splits > 1;
    const int qb = (p.q + (split ? 63 : 127)) / (split ? 64 : 128);
    const int count = p.b * H * qb * f.num_splits;
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < count;) {
        if (compute()) {
            char *memory = reinterpret_cast<char *>(&s.tile);
            if (split) {
                int bh = t / (qb * f.num_splits), q = t % qb, sp = (t / qb) % f.num_splits;
                if (p.q % 64 == 0 && f.seqlen_k % 128 == 0)
                    wam_whole_fa2::whole_split<FaSplit, false, false, false, true, true, false,
                                               true, false>(f, bh / H, bh % H, q, sp, f.num_splits,
                                                            memory);
                else
                    wam_whole_fa2::whole_split<FaSplit, false, false, false, false, true, false,
                                               true, false>(f, bh / H, bh % H, q, sp, f.num_splits,
                                                            memory);
            } else {
                int b = (t / qb) % p.b, h = t / (qb * p.b), q = t % qb;
                // B14实际Q16：只缩小本实例query tile，不跨候选拼接query。
                if (p.candidate && p.b == 14 && p.q == 16 && cross)
                    wam_whole_fa2::whole_standard_direct<FaCandidate, false, false, false, false,
                                                         false, true, false, false>(f, b, h, q, memory);
                else if (p.candidate && p.b == 14 && p.q == 16)
                    wam_whole_fa2::whole_standard<FaCandidate, false, false, false, false,
                                                  false, true, false, false>(f, b, h, q, memory);
                else if (cross && p.q % 128 == 0 && f.seqlen_k % 64 == 0)
                    wam_whole_fa2::whole_standard_direct<FaStandard, false, false, false, false,
                                                         true, true, false, false>(f, b, h, q, memory);
                else if (cross)
                    wam_whole_fa2::whole_standard_direct<FaStandard, false, false, false, false,
                                                         false, true, false, false>(f, b, h, q, memory);
                else if (p.q % 128 == 0 && f.seqlen_k % 64 == 0)
                    wam_whole_fa2::whole_standard<FaStandard, false, false, false, false, true,
                                                  true, false, false>(f, b, h, q, memory);
                else
                    wam_whole_fa2::whole_standard<FaStandard, false, false, false, false, false,
                                                  true, false, false>(f, b, h, q, memory);
            }
            asm volatile("cp.async.wait_all;" ::: "memory");
            wam_whole_fa2::whole_group_sync();
        }
        finish_tile(); // All eight warps join before shared tile storage is reused.
    }
    g.sync();
    // Combining KV splits remains a separate device stage, not another launch.
    const int combine_stage = cross ? 31 : 30;
    const int combine_count = split ? (p.b * H * p.q + 3) / 4 : 0;
    begin_stage(p, g, combine_stage);
    for (int t; (t = claim_tile(p, s, combine_stage)) < combine_count;) {
        if (compute()) {
            if (f.num_splits <= 2)
                fa2_combine_tile<1>(s, t);
            else if (f.num_splits <= 4)
                fa2_combine_tile<2>(s, t);
            else if (f.num_splits <= 8)
                fa2_combine_tile<3>(s, t);
            else if (f.num_splits <= 16)
                fa2_combine_tile<4>(s, t);
            else if (f.num_splits <= 32)
                fa2_combine_tile<5>(s, t);
            else if (f.num_splits <= 64)
                fa2_combine_tile<6>(s, t);
            else
                fa2_combine_tile<7>(s, t);
            wam_whole_fa2::whole_group_sync();
        }
        finish_tile();
    }
    g.sync();
}

// B14的attention阶段：一个CTA包含两个独立4-warp组，每组执行一个合法tile。
// 两组独享workspace/参数和barrier3/4；不跨候选共享private KV，也不跨阶段。
// 与单组不同，此时8个warp都做FA2；GEMM、Norm阶段仍按各自映射执行。
__device__ __noinline__ void attention_dual(const Params &p,Shared &s,cg::grid_group &g,int st,bool cross) {
    const int team=int(threadIdx.x)/128,tid=int(threadIdx.x)%128;
    const int prefix=int(p.info[0]);
    // 非16对齐的history尾部不进入G2；有限结构核验发现G2可改变BF16结果。
    int group_size=(prefix%16==0 && (prefix+32+63)/64==(prefix+16+63)/64)?2:1;
    for(int cfg=0;cfg<2;++cfg)
        for(int i=0;i<6;i+=2)
            if(p.rows[cfg*7+i]!=p.rows[cfg*7+i+1])group_size=1;
    const int groups=(7+group_size-1)/group_size;
    const int tasks=cross?p.b*H:2*groups*H;
    auto &f=s.group[team];
    if(tid==0) {
        f={};f.b=cross?p.b:1;f.h=f.h_k=H;f.h_h_k_ratio=1;f.d=f.d_rounded=HD;
        f.q_row_stride=f.o_row_stride=f.k_row_stride=f.v_row_stride=D;
        f.q_head_stride=f.o_head_stride=f.k_head_stride=f.v_head_stride=HD;
        f.seqlen_q_rounded=128;f.scale_softmax=p.attn_scale;f.scale_softmax_log2=p.attn_scale_log2;
        f.p_dropout=1.f;f.p_dropout_in_uint8_t=255;f.rp_dropout=1.f;f.scale_softmax_rp_dropout=p.attn_scale;
        f.window_size_left=f.window_size_right=-1;f.is_bf16=true;f.is_seqlens_k_cumulative=true;f.num_splits=1;
        if(cross) {
            f.q_ptr=bp(p,CQ);f.k_ptr=bp(p,CK);f.v_ptr=bp(p,CV);f.o_ptr=bp(p,CA);
            f.q_batch_stride=f.o_batch_stride=16*D;f.k_batch_stride=f.v_batch_stride=int64_t(p.c)*D;
            f.seqlen_q=16;f.seqlen_k=p.c;f.seqlen_k_rounded=(p.c+127)/128*128;f.softmax_lse_ptr=p.fa_lse;
        } else {
            f.prefix_k=p.pk;f.prefix_v=p.pv;f.current_k=bp(p,SKR);f.current_v=bp(p,SV);
            f.k_ptr=p.pk;f.v_ptr=p.pv;f.valid_slots=p.slots;f.prefix_length=prefix;f.candidates=7;
            f.pk_batch=f.pv_batch=int64_t(p.pool)*D;f.pk_row=f.pv_row=D;f.pk_head=f.pv_head=HD;
            f.ck_batch=f.cv_batch=16*D;f.ck_row=f.cv_row=D;f.ck_head=f.cv_head=HD;
            f.q_batch_stride=f.o_batch_stride=f.k_batch_stride=f.v_batch_stride=0;
        }
        if(blockIdx.x==0 && team==0){const int o=cross?4:0;p.fa_meta[o]=cross?p.c:prefix+16;p.fa_meta[o+1]=1;p.fa_meta[o+2]=16;p.fa_meta[o+3]=FA_SMEM;}
    }
    begin_stage(p,g,st);
    for(int task=2*int(blockIdx.x)+team;task<tasks;task+=2*int(gridDim.x)) {
        if(cross) {
            wam_whole_fa2::whole_direct_dual<FaCandidate,false,false,false,false,false,true,false,false>(
                f,task%p.b,task/p.b,0,s.tile.fa2_dual[team]);
        } else {
            const int group=task%(2*groups),head=task/(2*groups),cfg=group/groups;
            const int local=group%groups,first=cfg*7+local*group_size,count=min(group_size,7-local*group_size);
            if(tid==0) {
                f.group_base=first;f.group_cfg=int(p.rows[first]);f.q_ptr=bp(p,SQR)+first*16*D;f.o_ptr=bp(p,SA)+first*16*D;
                f.softmax_lse_ptr=p.fa_lse+first*H*16;f.seqlen_q=count*16;f.seqlen_k=prefix+count*16;f.seqlen_k_rounded=(f.seqlen_k+127)/128*128;
            }
            wam_whole_fa2::dual_group_sync();
            wam_whole_fa2::whole_grouped_dual<FaCandidate,false,false,false,false,false,true,false,false>(
                f,0,head,0,s.tile.fa2_dual[team]);
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        wam_whole_fa2::dual_group_sync();
    }
    g.sync();
}
__device__ __forceinline__ void attention(const Params &p,Shared &s,cg::grid_group &g,int st,bool cross) {
    if(p.candidate && p.b==14 && p.q==16)attention_dual(p,s,g,st,cross);
    else independent_attention(p,s,g,st,cross);
}
