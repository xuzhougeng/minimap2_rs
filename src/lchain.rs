use crate::seeds::Anchor;

#[inline]
fn qpos(a: &Anchor) -> i32 { (a.y & 0xffffffff) as i32 }
#[inline]
fn qspan(a: &Anchor) -> i32 { ((a.y >> 32) & 0xff) as i32 }
#[inline]
fn rpos(a: &Anchor) -> i32 { (a.x & 0xffffffff) as i32 }
#[inline]
fn rev(a: &Anchor) -> bool { (a.x >> 63) != 0 }
#[inline]
fn rid(a: &Anchor) -> i32 { ((a.x >> 32) & 0x7fffffff) as i32 }

#[inline]
fn mg_log2(x: i32) -> f32 { if x <= 1 { 0.0 } else { (x as f32).ln() / std::f32::consts::LN_2 } }

fn comput_sc(ai: &Anchor, aj: &Anchor, max_dist_x: i32, max_dist_y: i32, bw: i32, chn_pen_gap: f32, chn_pen_skip: f32) -> Option<i32> {
    let dq = qpos(ai) - qpos(aj);
    if dq <= 0 || dq > max_dist_x { return None; }
    let dr = rpos(ai) - rpos(aj);
    // Match minimap2 semantics: require progress on reference and constrain query gap by max_dist_y
    if dr == 0 || dq > max_dist_y { return None; }
    let dd = (dr - dq).abs();
    if dd > bw { return None; }
    let dg = dr.min(dq);
    let q_span = qspan(aj);
    let mut sc = q_span.min(dg);
    if dd != 0 || dg > q_span {
        let lin_pen = chn_pen_gap * (dd as f32) + chn_pen_skip * (dg as f32);
        let log_pen = if dd >= 1 { mg_log2(dd + 1) } else { 0.0 };
        sc -= (lin_pen + 0.5 * log_pen) as i32;
    }
    Some(sc)
}

#[derive(Clone)]
pub struct ChainParams {
    pub max_dist_x: i32,
    pub max_dist_y: i32,
    pub bw: i32,
    pub max_chain_iter: i32,
    pub min_chain_score: i32,
    pub min_cnt: i32,
    pub chn_pen_gap: f32,
    pub chn_pen_skip: f32,
    pub max_chain_skip: i32,
    pub max_drop: i32,
    // rescue/long-join
    pub bw_long: i32,
    pub rmq_rescue_size: i32,
    pub rmq_rescue_ratio: f32,
}

pub fn chain_dp(anchors: &[Anchor], p: &ChainParams) -> Vec<usize> {
    let (chains, _scores) = chain_dp_all(anchors, p);
    chains.into_iter().next().unwrap_or_default()
}

pub fn chain_dp_all(anchors: &[Anchor], p: &ChainParams) -> (Vec<Vec<usize>>, Vec<i32>) {
    let n = anchors.len();
    if n == 0 { return (Vec::new(), Vec::new()); }
    // adjust distances per minimap2
    let mut max_dist_x = p.max_dist_x;
    let mut max_dist_y = p.max_dist_y;
    if max_dist_x < p.bw { max_dist_x = p.bw; }
    if max_dist_y < p.bw { max_dist_y = p.bw; }
    // DP arrays
    let mut f = vec![0i32; n];
    let mut v = vec![0i32; n];
    let mut t = vec![0i32; n];
    let mut pprev = vec![-1isize; n];
    let mut st: usize = 0;
    let mut max_ii: isize = -1;
    for i in 0..n {
        while st < i && (rid(&anchors[st]) != rid(&anchors[i]) || rev(&anchors[st]) != rev(&anchors[i]) || rpos(&anchors[i]) > rpos(&anchors[st]) + max_dist_x) { st += 1; }
        let mut max_j: isize = -1;
        let mut max_f = qspan(&anchors[i]);
        let start_j = if i as i32 - p.max_chain_iter > st as i32 { (i as i32 - p.max_chain_iter) as usize } else { st };
        let mut n_skip = 0i32;
        for j in (start_j..i).rev() {
            if rid(&anchors[j]) != rid(&anchors[i]) || rev(&anchors[j]) != rev(&anchors[i]) { continue; }
            if let Some(sc0) = comput_sc(&anchors[i], &anchors[j], max_dist_x, max_dist_y, p.bw, p.chn_pen_gap, p.chn_pen_skip) {
                let sc = sc0 + f[j];
                if sc > max_f { max_f = sc; max_j = j as isize; if n_skip > 0 { n_skip -= 1; } }
                else if t[j] == i as i32 { n_skip += 1; if n_skip > p.max_chain_skip { break; } }
                if pprev[j] >= 0 { t[pprev[j] as usize] = i as i32; }
            }
        }
        f[i] = max_f; pprev[i] = max_j;
        v[i] = if max_j >= 0 && v[max_j as usize] > max_f { v[max_j as usize] } else { max_f };
    }
    // backtrack like minimap2
    // z: indices with f[i] >= min_chain_score, sorted by f asc
    let mut z: Vec<(i32, usize)> = Vec::new();
    for i in 0..n { if f[i] > 0 { z.push((f[i], i)); } }
    if z.is_empty() { return (Vec::new(), Vec::new()); }
    z.sort_unstable_by_key(|x| x.0);
    t.fill(0);
    let mut n_v = 0usize; let mut n_u = 0usize;
    // first pass: count total v and number of chains
    for k in (0..z.len()).rev() {
        let i0 = z[k].1;
        if t[i0] != 0 { continue; }
        // walk back until drop exceeds max_drop (use p.bw like minimap2 default)
        let mut i = i0 as isize;
        let mut end_i: isize = -1;
        let mut max_s = 0i32; let mut max_i = i;
        if i >= 0 && t[i as usize] == 0 {
            loop {
                t[i as usize] = 2;
                end_i = pprev[i as usize];
                let s = if end_i < 0 { z[k].0 } else { z[k].0 - f[end_i as usize] };
                if s > max_s { max_s = s; max_i = end_i; } else if max_s - s > p.max_drop { break; }
                if !(i >= 0 && t[i as usize] == 0 && end_i >= 0) { break; }
                i = end_i;
            }
            let mut ii = i0 as isize;
            while ii >= 0 && ii != end_i { t[ii as usize] = 0; ii = pprev[ii as usize]; }
        }
        // now collect length and score threshold
        let mut len0 = n_v;
        let mut i = i0 as isize; let end_i = max_i;
        while i >= 0 && i != end_i { n_v += 1; t[i as usize] = 1; i = pprev[i as usize]; }
        let sc = if i < 0 { z[k].0 } else { z[k].0 - f[i as usize] };
        if sc >= p.min_chain_score && n_v > len0 && (n_v - len0) as i32 >= p.min_cnt { n_u += 1; } else { n_v = len0; }
    }
    // second pass: populate chains
    let mut chains: Vec<Vec<usize>> = Vec::with_capacity(n_u);
    let mut scores: Vec<i32> = Vec::with_capacity(n_u);
    t.fill(0);
    for k in (0..z.len()).rev() {
        let i0 = z[k].1;
        if t[i0] != 0 { continue; }
        // mg_chain_bk_end
        let mut i = i0 as isize;
        let mut end_i: isize = -1;
        let mut max_s = 0i32; let mut max_i = i;
        if i >= 0 && t[i as usize] == 0 {
            loop {
                t[i as usize] = 2;
                end_i = pprev[i as usize];
                let s = if end_i < 0 { z[k].0 } else { z[k].0 - f[end_i as usize] };
                if s > max_s { max_s = s; max_i = end_i; } else if max_s - s > p.max_drop { break; }
                if !(i >= 0 && t[i as usize] == 0 && end_i >= 0) { break; }
                i = end_i;
            }
            let mut ii = i0 as isize;
            while ii >= 0 && ii != end_i { t[ii as usize] = 0; ii = pprev[ii as usize]; }
        }
        // extract chain
        let mut v_idxs: Vec<usize> = Vec::new();
        let mut i = i0 as isize; let end_i = max_i;
        while i >= 0 && i != end_i { v_idxs.push(i as usize); t[i as usize] = 1; i = pprev[i as usize]; }
        let sc = if i < 0 { z[k].0 } else { z[k].0 - f[i as usize] };
        if sc >= p.min_chain_score && v_idxs.len() as i32 >= p.min_cnt {
            v_idxs.reverse();
            scores.push(sc);
            chains.push(v_idxs);
        }
    }
    // Fallback: if no chains pass backtracking thresholds, build the single best chain greedily from prev[]
    if chains.is_empty() {
        if let Some((best_i, _)) = (0..n).map(|i| (i, f[i])).max_by_key(|&(_, sc)| sc) {
            let mut v_idxs: Vec<usize> = Vec::new();
            let mut i = best_i as isize;
            while i >= 0 { v_idxs.push(i as usize); i = pprev[i as usize]; }
            v_idxs.reverse();
            if !v_idxs.is_empty() {
                chains.push(v_idxs);
                scores.push(v[best_i]);
            }
        }
    }
    // sort chains by score desc, then by qstart asc, then tstart asc to match downstream behavior
    sort_chains_stable(anchors, chains, scores)
}

#[inline]
fn chain_qrange(anchors: &[Anchor], chain: &[usize]) -> (i32,i32) {
    let mut qs = i32::MAX; let mut qe = -1;
    for &i in chain {
        let a = &anchors[i];
        let s = qpos(a) - (qspan(a) - 1);
        let e = qpos(a) + 1;
        if s < qs { qs = s; } if e > qe { qe = e; }
    }
    (qs.max(0), qe)
}

#[inline]
fn chain_trange(anchors: &[Anchor], chain: &[usize]) -> (i32,i32) {
    let mut ts = i32::MAX; let mut te = -1;
    for &i in chain {
        let a = &anchors[i];
        let s = rpos(a) - (qspan(a) - 1);
        let e = rpos(a) + 1;
        if s < ts { ts = s; } if e > te { te = e; }
    }
    (ts.max(0), te)
}

pub fn sort_chains_stable(anchors: &[Anchor], mut chains: Vec<Vec<usize>>, mut scores: Vec<i32>) -> (Vec<Vec<usize>>, Vec<i32>) {
    // stable sort by (score desc, qstart asc, tstart asc)
    let mut idxs: Vec<usize> = (0..chains.len()).collect();
    idxs.sort_by(|&i,&j| {
        let si = scores[i]; let sj = scores[j];
        if si != sj { return sj.cmp(&si); }
        let (qi, _) = chain_qrange(anchors, &chains[i]);
        let (qj, _) = chain_qrange(anchors, &chains[j]);
        if qi != qj { return qi.cmp(&qj); }
        let (ti, _) = chain_trange(anchors, &chains[i]);
        let (tj, _) = chain_trange(anchors, &chains[j]);
        ti.cmp(&tj)
    });
    let chains2 = idxs.iter().map(|&i| chains[i].clone()).collect();
    let scores2 = idxs.iter().map(|&i| scores[i]).collect();
    (chains2, scores2)
}

pub fn select_primary_secondary(anchors: &[Anchor], chains: &[Vec<usize>], scores: &[i32], mask_level: f32) -> Vec<bool> {
    // returns is_primary flags sorted by input order (assume chains already sorted by score desc)
    let mut primaries: Vec<(i32,(i32,i32))> = Vec::new(); // (score, qrange)
    let mut is_primary = vec![true; chains.len()];
    for (ci, chain) in chains.iter().enumerate() {
        let (qs, qe) = chain_qrange(anchors, chain);
        let mut overlapped = false;
        for &(_s,(pqs,pqe)) in &primaries {
            let ov = (qe.min(pqe) - qs.max(pqs)).max(0) as f32;
            let len = (qe - qs).max(1) as f32;
            if ov / len >= mask_level { overlapped = true; break; }
        }
        if overlapped { is_primary[ci] = false; } else { primaries.push((scores[ci], (qs,qe))); }
    }
    is_primary
}

pub fn select_and_filter_chains(anchors: &[Anchor], chains: &[Vec<usize>], scores: &[i32], mask_level: f32, pri_ratio: f32, best_n: usize) -> (Vec<Vec<usize>>, Vec<i32>, Vec<bool>, i32, i32) {
    // assumes chains sorted by score desc
    if chains.is_empty() { return (Vec::new(), Vec::new(), Vec::new(), 0, 0); }
    let (chains, scores) = sort_chains_stable(anchors, chains.to_vec(), scores.to_vec());
    let is_primary = select_primary_secondary(anchors, &chains, &scores, mask_level);
    let mut out_chains = Vec::new();
    let mut out_scores = Vec::new();
    let mut out_is_primary = Vec::new();
    let s1 = scores[0];
    let mut s2 = 0;
    let mut sec_kept = 0usize;
    for (i, chain) in chains.iter().enumerate() {
        if i == 0 {
            out_chains.push(chain.clone()); out_scores.push(scores[i]); out_is_primary.push(true);
        } else {
            if !is_primary[i] { continue; }
            if (scores[i] as f32) >= pri_ratio * (s1 as f32) {
                if sec_kept < best_n { out_chains.push(chain.clone()); out_scores.push(scores[i]); out_is_primary.push(false); sec_kept += 1; }
            }
            if s2 == 0 { s2 = scores[i]; }
        }
    }
    (out_chains, out_scores, out_is_primary, s1, s2)
}

pub fn merge_adjacent_chains(anchors: &[Anchor], chains: &[Vec<usize>]) -> Vec<Vec<usize>> {
    // sort chains by query start and merge if adjacent on same rid/strand and touching
    let mut items: Vec<(i32, usize)> = chains.iter().enumerate().map(|(i,ch)| {
        let (qs, _qe) = chain_qrange(anchors, ch); (qs, i)
    }).collect();
    items.sort_unstable_by_key(|x| x.0);
    let mut merged: Vec<Vec<usize>> = Vec::new();
    for (_qs, idx) in items {
        let ch = &chains[idx];
        if merged.is_empty() { merged.push(ch.clone()); continue; }
        // last merged
        let last = merged.last_mut().unwrap();
        let a_last = anchors[*last.last().unwrap()];
        let a_first = anchors[*ch.first().unwrap()];
        let same = rid(&a_last) == rid(&a_first) && rev(&a_last) == rev(&a_first);
        let (_, last_qe) = chain_qrange(anchors, last);
        let (ch_qs, _) = chain_qrange(anchors, ch);
        if same && ch_qs <= last_qe { // touching or overlapping: merge
            last.extend_from_slice(ch);
        } else {
            merged.push(ch.clone());
        }
    }
    merged
}

pub fn merge_adjacent_chains_with_gap(anchors: &[Anchor], chains: &[Vec<usize>], max_gap_q: i32, max_gap_t: i32) -> Vec<Vec<usize>> {
    let mut items: Vec<(i32, usize)> = chains.iter().enumerate().map(|(i,ch)| {
        let (qs, _qe) = chain_qrange(anchors, ch); (qs, i)
    }).collect();
    items.sort_unstable_by_key(|x| x.0);
    let mut merged: Vec<Vec<usize>> = Vec::new();
    for (_qs, idx) in items {
        let ch = &chains[idx];
        if merged.is_empty() { merged.push(ch.clone()); continue; }
        let last = merged.last_mut().unwrap();
        let a_last = anchors[*last.last().unwrap()];
        let a_first = anchors[*ch.first().unwrap()];
        let same = rid(&a_last) == rid(&a_first) && rev(&a_last) == rev(&a_first);
        let (_last_qs, last_qe) = chain_qrange(anchors, last);
        let (ch_qs, _ch_qe) = chain_qrange(anchors, ch);
        let (_last_ts, last_te) = chain_trange(anchors, last);
        let (ch_ts, _ch_te) = chain_trange(anchors, ch);
        let q_gap = ch_qs - last_qe;
        let t_gap = ch_ts - last_te;
        if same && q_gap >= 0 && t_gap >= 0 && q_gap <= max_gap_q && t_gap <= max_gap_t { // within gap thresholds: merge
            last.extend_from_slice(ch);
        } else {
            merged.push(ch.clone());
        }
    }
    merged
}

pub fn chain_query_coverage(anchors: &[Anchor], chain: &[usize]) -> i32 {
    let (qs, qe) = chain_qrange(anchors, chain);
    (qe - qs).max(0)
}

pub fn rescue_long_join(anchors: &[Anchor], chains: &[Vec<usize>], scores: &[i32], p: &ChainParams, qlen: i32) -> (Vec<Vec<usize>>, Vec<i32>) {
    if chains.is_empty() { return (chains.to_vec(), scores.to_vec()); }
    let best_cov = chain_query_coverage(anchors, &chains[0]);
    let uncovered = (qlen - best_cov).max(0);
    let rescue = uncovered > p.rmq_rescue_size || (best_cov as f32) < (qlen as f32) * (1.0 - p.rmq_rescue_ratio);
    if !rescue { return (chains.to_vec(), scores.to_vec()); }
    let mut p2 = p.clone();
    p2.bw = p.bw_long;
    chain_dp_all(anchors, &p2)
}
