use crate::seeds::{Anchor, collect_query_minimizers};
use crate::index::Index;

pub struct PafRecord<'a> {
    pub qname: &'a str,
    pub qlen: u32,
    pub qstart: u32,
    pub qend: u32,
    pub strand: char,
    pub tname: &'a str,
    pub tlen: u32,
    pub tstart: u32,
    pub tend: u32,
    pub nm: u32,
    pub blen: u32,
    pub mapq: u8,
    // tags
    pub tp: char,
    pub cm: u32,
    pub s1: u32,
    pub s2: u32,
    pub dv: f32,
    pub rl: u32,
}

#[inline]
fn qpos(a: &Anchor) -> i32 { (a.y & 0xffffffff) as i32 }
#[inline]
fn qspan(a: &Anchor) -> i32 { ((a.y >> 32) & 0xff) as i32 }
#[inline]
fn rpos(a: &Anchor) -> i32 { (a.x & 0xffffffff) as i32 }
#[inline]
fn rev(a: &Anchor) -> bool { (a.x >> 63) != 0 }

fn banded_edit_distance(q: &[u8], r: &[u8], band: usize) -> (usize, usize) {
    // Simple banded Levenshtein: returns (edits, aligned_len)
    let n = q.len(); let m = r.len();
    if n == 0 || m == 0 { return (n.max(m), n.max(m)); }
    let b = band;
    let inf = n + m + 1;
    // dp rows of size 2*b+1 centered around diagonal
    let width = 2*b+1;
    let mut prev = vec![inf; width];
    let mut curr = vec![inf; width];
    // offset diag index k = j - i, map to idx = k + b
    let idx = |i: isize, j: isize| -> Option<usize> {
        let k = j - i; if k.abs() as usize > b { return None; }
        Some((k + b as isize) as usize)
    };
    prev[b] = 0; // i=0,j=0
    for i in 0..=n {
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(inf);
        let j_min = if i > b { i - b } else { 0 };
        let j_max = (i + b).min(m);
        for j in j_min..=j_max {
            let ii = i as isize; let jj = j as isize;
            let mut best = inf;
            // deletion (from q)
            if i>0 { if let Some(k) = idx(ii-1, jj) { best = best.min(prev[k] + 1); } }
            // insertion
            if j>0 { if let Some(k) = idx(ii, jj-1) { best = best.min(curr[k] + 1); } }
            // match/mismatch
            if i>0 && j>0 { if let Some(k) = idx(ii-1, jj-1) {
                let cost = if q[i-1].to_ascii_uppercase()==r[j-1].to_ascii_uppercase() {0} else {1};
                best = best.min(prev[k] + cost);
            }}
            if let Some(k) = idx(ii, jj) { curr[k] = best; }
        }
    }
    let end = idx(n as isize, m as isize);
    if let Some(k) = end {
        let edits = curr[k];
        (edits, n.max(m))
    } else {
        // outside band: fall back to maximum possible edits
        (n.max(m), n.max(m))
    }
}

fn estimate_dv_by_mismatch(q: &[u8], r: &[u8]) -> f32 {
    if q.is_empty() || r.is_empty() { return 0.0; }
    let n = q.len().min(r.len());
    let mut mis = 0usize;
    for i in 0..n { if q[i].to_ascii_uppercase() != r[i].to_ascii_uppercase() { mis += 1; } }
    (mis as f32) / (n as f32)
}

fn end_extend(idx: &Index, qseq: &[u8], rid: usize, _strand: char, mut qs: i32, mut qe: i32, mut ts: i32, mut te: i32, max_ext: i32) -> (i32,i32,i32,i32) {
    let tlen = idx.seq[rid].len as i32;
    let qlen = qseq.len() as i32;
    // left extension
    let mut ext = 0;
    while ext < max_ext && qs > 0 && ts > 0 {
        let qb = qseq[(qs-1) as usize].to_ascii_uppercase();
        let rb = idx.get_ref_subseq(rid, ts-1, ts)[0].to_ascii_uppercase();
        if qb != rb { break; }
        qs -= 1; ts -= 1; ext += 1;
    }
    // right extension
    ext = 0;
    while ext < max_ext && qe < qlen && te < tlen {
        let qb = qseq[qe as usize].to_ascii_uppercase();
        let rb = idx.get_ref_subseq(rid, te, te+1)[0].to_ascii_uppercase();
        if qb != rb { break; }
        qe += 1; te += 1; ext += 1;
    }
    (qs, qe, ts, te)
}

#[inline]
fn revcomp(seq: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seq.len());
    for &b in seq.iter().rev() {
        out.push(match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        });
    }
    out
}

pub fn paf_from_chain<'a>(idx: &'a Index, anchors: &[Anchor], chain: &[usize], qname: &'a str, qseq: &[u8]) -> Option<PafRecord<'a>> {
    paf_from_chain_with_primary(idx, anchors, chain, qname, qseq, true)
}

pub fn paf_from_chain_with_primary<'a>(idx: &'a Index, anchors: &[Anchor], chain: &[usize], qname: &'a str, qseq: &[u8], is_primary: bool) -> Option<PafRecord<'a>> {
    if chain.is_empty() { return None; }
    let strand = if rev(&anchors[chain[0]]) { '-' } else { '+' };
    let mut qs = i32::MAX; let mut qe = -1;
    let mut ts = i32::MAX; let mut te = -1;
    let mut cm = 0u32;
    for &i in chain {
        let a = &anchors[i];
        cm += 1;
        let s = qpos(a) - (qspan(a) - 1);
        let e = qpos(a) + 1;
        if s < qs { qs = s; } if e > qe { qe = e; }
        let rs = rpos(a) - (qspan(a) - 1);
        let re = rpos(a) + 1;
        if rs < ts { ts = rs; } if re > te { te = re; }
    }
    if qs < 0 { qs = 0; }
    if ts < 0 { ts = 0; }
    let rid0 = ((anchors[chain[0]].x >> 32) & 0x7fffffff) as usize;
    let tname = idx.seq[rid0].name.as_deref().unwrap_or("*");
    let tlen = idx.seq[rid0].len;
    // avoid end-extension to keep boundaries aligned with chaining like minimap2 pre-alignment
    let (qs2, qe2, ts2, te2) = (qs, qe, ts, te);
    let mlen = (qe2 - qs2).max(0) as u32; // approximate matches along query span
    let blen = (te2 - ts2).max(0) as u32; // alignment block length along target span
    // dv estimate following mm_est_err logic using query minimizer positions
    let mv = collect_query_minimizers(qseq, idx.w as usize, idx.k as usize);
    let mut mini_pos: Vec<i32> = Vec::with_capacity(mv.len());
    let mut sum_k: u64 = 0;
    for m in &mv { mini_pos.push(((m.rid_pos_strand >> 1) & 0xffffffff) as i32); sum_k += (m.key_span & 0xff) as u64; }
    let avg_k: f32 = if !mv.is_empty() { (sum_k as f32) / (mv.len() as f32) } else { idx.k as f32 };
    // prepare chain query positions on forward strand
    let qpos_fwd = |a: &Anchor| -> i32 {
        let qp = qpos(a);
        let qs = qspan(a);
        if rev(a) { (qseq.len() as i32) - 1 - (qp + 1 - qs) } else { qp }
    };
    let mut chain_qs_fwd: Vec<i32> = Vec::with_capacity(chain.len());
    if strand == '-' {
        for &i in chain.iter().rev() { chain_qs_fwd.push(qpos_fwd(&anchors[i])); }
    } else {
        for &i in chain.iter() { chain_qs_fwd.push(qpos_fwd(&anchors[i])); }
    }
    // find start index in minimizer list
    let mut dv = 0.0f32;
    if !mini_pos.is_empty() && !chain_qs_fwd.is_empty() {
        // binary search for first equal position
        let first = chain_qs_fwd[0];
        if let Ok(mut st) = mini_pos.binary_search(&first) {
            // move st to the first occurrence if duplicates
            while st > 0 && mini_pos[st-1] == first { st -= 1; }
            let mut j = st;
            let mut k = 1usize;
            let mut en = st;
            let mut n_match = 1i32;
            while j + 1 < mini_pos.len() && k < chain_qs_fwd.len() {
                j += 1;
                if mini_pos[j] == chain_qs_fwd[k] { n_match += 1; en = j; k += 1; }
            }
            let mut n_tot: i32 = ((en - st) + 1) as i32;
            // edge adjustment exactly like minimap2 (using printed coordinates)
            let r_qs_final: i32 = if strand == '-' { qseq.len() as i32 - qe2 } else { qs2 };
            let r_qe_final: i32 = if strand == '-' { qseq.len() as i32 - qs2 } else { qe2 };
            let r_rs: i32 = ts2; let r_re: i32 = te2;
            if r_qs_final > avg_k as i32 && r_rs > avg_k as i32 { n_tot += 1; }
            if (qseq.len() as i32 - r_qe_final) > avg_k as i32 && (tlen as i32 - r_re) > avg_k as i32 { n_tot += 1; }
            let frac = (n_match as f32) / (n_tot as f32);
            if frac >= 1.0 { dv = 0.0; }
            else { dv = 1.0 - frac.powf(1.0 / avg_k.max(1.0)); }
        }
    }
    let rec = PafRecord{
        qname,
        qlen: qseq.len() as u32,
        qstart: qs2 as u32,
        qend: qe2 as u32,
        strand,
        tname,
        tlen,
        tstart: ts2 as u32,
        tend: te2 as u32,
        nm: mlen,
        blen,
        mapq: 60,
        tp: if is_primary {'P'} else {'S'},
        cm,
        s1: 0,
        s2: 0,
        dv,
        rl: 0,
    };
    Some(rec)
}

pub fn write_paf(rec: &PafRecord) -> String {
    let (qs, qe) = if rec.strand == '-' {
        (rec.qlen - rec.qend, rec.qlen - rec.qstart)
    } else { (rec.qstart, rec.qend) };
    format!(
        "{q}\t{ql}\t{qs}\t{qe}\t{s}\t{t}\t{tl}\t{ts}\t{te}\t{nm}\t{bl}\t{mq}\ttp:A:{tp}\tcm:i:{cm}\ts1:i:{s1}\ts2:i:{s2}\tdv:f:{dv:.4}\trl:i:{rl}",
        q=rec.qname, ql=rec.qlen, qs=qs, qe=qe,
        s=rec.strand, t=rec.tname, tl=rec.tlen, ts=rec.tstart, te=rec.tend,
        nm=rec.nm, bl=rec.blen, mq=rec.mapq,
        tp=rec.tp, cm=rec.cm, s1=rec.s1, s2=rec.s2,
        dv=rec.dv, rl=rec.rl,
    )
}

pub fn write_paf_many_with_scores<'a>(idx: &'a Index, anchors: &[Anchor], chains: &[Vec<usize>], top_s1: i32, top_s2: i32, qname: &'a str, qseq: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    for (ci, chain) in chains.iter().enumerate() {
        if let Some(mut rec) = paf_from_chain_with_primary(idx, anchors, chain, qname, qseq, ci == 0) {
            rec.s1 = top_s1.max(0) as u32;
            rec.s2 = top_s2.max(0) as u32;
            out.push(write_paf(&rec));
        }
    }
    out
}

pub fn write_paf_many<'a>(idx: &'a Index, anchors: &[Anchor], chains: &[Vec<usize>], qname: &'a str, qseq: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    for (ci, chain) in chains.iter().enumerate() {
        if let Some(rec) = paf_from_chain_with_primary(idx, anchors, chain, qname, qseq, ci == 0) {
            out.push(write_paf(&rec));
        }
    }
    out
}
