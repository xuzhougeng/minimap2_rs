use crate::index::{Index, Occurrences};
use crate::sketch::{sketch_sequence, Minimizer};

#[derive(Clone, Copy, Debug)]
pub struct Anchor { pub x: u64, pub y: u64 }

pub fn collect_query_minimizers(seq: &[u8], w: usize, k: usize) -> Vec<Minimizer> {
    let mut v = Vec::new();
    sketch_sequence(seq, w, k, 0, false, &mut v)
    ;v
}

pub fn filter_query_minimizers(mv: &mut Vec<Minimizer>, q_occ_max: i32, q_occ_frac: f32) {
    if mv.len() == 0 || (q_occ_frac <= 0.0) || (q_occ_max <= 0) { return; }
    if mv.len() as i32 <= q_occ_max { return; }
    let mut keys: Vec<(u64, usize)> = Vec::with_capacity(mv.len());
    for (i, m) in mv.iter().enumerate() {
        keys.push(((m.key_span >> 8), i));
    }
    keys.sort_unstable_by_key(|x| x.0);
    let mut keep = vec![true; mv.len()];
    let mut st = 0usize; let n = keys.len();
    let cutoff = (mv.len() as f32 * q_occ_frac) as usize;
    for i in 1..=n {
        if i == n || keys[i].0 != keys[st].0 {
            let cnt = i - st;
            if cnt as i32 > q_occ_max && cnt > cutoff {
                for j in st..i { keep[keys[j].1] = false; }
            }
            st = i;
        }
    }
    let mut j = 0usize;
    for i in 0..mv.len() { if keep[i] { mv[j] = mv[i]; j += 1; } }
    mv.truncate(j);
}

pub fn build_anchors(idx: &Index, mv: &[Minimizer], qlen: i32) -> Vec<Anchor> {
    build_anchors_filtered(idx, mv, qlen, i32::MAX)
}

pub fn build_anchors_filtered(idx: &Index, mv: &[Minimizer], qlen: i32, mid_occ: i32) -> Vec<Anchor> {
    let mut a: Vec<Anchor> = Vec::new();
    for m in mv {
        let minier = m.key_span >> 8; // hashed kmer key
        if let Some(occ) = idx.get(minier) {
            match occ {
                Occurrences::Single(r) => {
                    push_anchor(&mut a, r, m, qlen);
                }
                Occurrences::Multi(slice) => {
                    if slice.len() as i32 > mid_occ { continue; }
                    for &r in slice { push_anchor(&mut a, r, m, qlen); }
                }
            }
        }
    }
    a.sort_by(|p,q| if p.x==q.x { p.y.cmp(&q.y) } else { p.x.cmp(&q.x) });
    a
}

#[inline]
fn push_anchor(out: &mut Vec<Anchor>, r: u64, m: &Minimizer, qlen: i32) {
    let rid = (r >> 32) & 0xffffffff;
    let rpos = ((r >> 1) & 0xffffffff) as i32; // last pos
    let rstrand = (r & 1) as i32;
    let qpos = ((m.rid_pos_strand >> 1) & 0xffffffff) as i32;
    let qstrand = (m.rid_pos_strand & 1) as i32;
    let qspan = (m.key_span & 0xff) as i32;
    let forward = rstrand == qstrand;
    let x = if forward { ((rid as u64) << 32) | (rpos as u64) } else { (1u64<<63) | ((rid as u64) << 32) | (rpos as u64) };
    let y = if forward {
        ((qspan as u64) << 32) | (qpos as u64)
    } else {
        let qp = (qlen - (qpos + 1 - qspan) - 1) as u64;
        ((qspan as u64) << 32) | qp
    };
    out.push(Anchor{ x, y });
}
