use crate::nt4::nt4;

#[inline]
fn hash64(mut key: u64, mask: u64) -> u64 {
    key = (!key).wrapping_add(key << 21) & mask;
    key ^= key >> 24;
    key = (key + (key << 3) + (key << 8)) & mask;
    key ^= key >> 14;
    key = (key + (key << 2) + (key << 4)) & mask;
    key ^= key >> 28;
    key = (key + (key << 31)) & mask;
    key
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Minimizer {
    pub key_span: u64,           // key<<8 | span
    pub rid_pos_strand: u64,     // rid<<32 | last_pos<<1 | strand
}

struct TinyQueue { front: usize, count: usize, a: [i32; 32] }
impl TinyQueue {
    fn new() -> Self { Self { front: 0, count: 0, a: [0; 32] } }
    #[inline] fn clear(&mut self) { self.front = 0; self.count = 0; }
    #[inline] fn push(&mut self, x: i32) { let idx = (self.count + self.front) & 0x1f; self.a[idx] = x; self.count += 1; }
    #[inline] fn shift(&mut self) -> i32 { if self.count == 0 { return -1; } let x = self.a[self.front]; self.front = (self.front + 1) & 0x1f; self.count -= 1; x }
}

pub fn sketch_sequence(seq: &[u8], w: usize, k: usize, rid: u32, is_hpc: bool, out: &mut Vec<Minimizer>) {
    assert!(!seq.is_empty());
    assert!(w > 0 && w < 256);
    assert!(k > 0 && k <= 28);

    let shift1: u64 = 2 * (k as u64 - 1);
    let mask: u64 = (1u64 << (2 * k)) - 1;
    let mut kmer = [0u64; 2];

    let mut l: i32 = 0;
    let mut buf_pos: usize = 0;
    let mut min_pos: usize = 0;
    let mut kmer_span: i32 = 0;
    let mut buf: Vec<(u64, u64)> = vec![(u64::MAX, u64::MAX); w];
    let mut min: (u64, u64) = (u64::MAX, u64::MAX);

    let mut tq = TinyQueue::new();

    for i in 0..seq.len() {
        let c = nt4(seq[i]) as i32;
        let mut info = (u64::MAX, u64::MAX);
        if c < 4 {
            if is_hpc {
                // compute homopolymer run length starting at i
                let mut skip_len = 1usize;
                if i + 1 < seq.len() && (nt4(seq[i + 1]) as i32) == c {
                    let mut t = i + 2;
                    while t < seq.len() && (nt4(seq[t]) as i32) == c { t += 1; }
                    skip_len = t - i;
                }
                tq.push(skip_len as i32);
                kmer_span += skip_len as i32;
                if tq.count as i32 > k as i32 { kmer_span -= tq.shift(); }
            } else {
                kmer_span = if l + 1 < k as i32 { l + 1 } else { k as i32 };
            }
            kmer[0] = ((kmer[0] << 2) | (c as u64)) & mask;                  // forward
            kmer[1] = (kmer[1] >> 2) | (((3 ^ c) as u64) << shift1);         // reverse
            if kmer[0] != kmer[1] { // skip symmetric k-mers
                let z = if kmer[0] < kmer[1] { 0 } else { 1 };
                l += 1;
                if l >= k as i32 && kmer_span < 256 {
                    let key_span = (hash64(kmer[z], mask) << 8) | (kmer_span as u64);
                    let rid_pos_strand = ((rid as u64) << 32) | ((i as u64) << 1) | (z as u64);
                    info = (key_span, rid_pos_strand);
                }
            }
        } else {
            l = 0; tq.clear(); kmer_span = 0;
        }
        buf[buf_pos] = info;
        if l == (w as i32 + k as i32 - 1) && min.0 != u64::MAX {
            for j in (buf_pos + 1)..w { if min.0 == buf[j].0 && buf[j].1 != min.1 { out.push(Minimizer{ key_span: buf[j].0, rid_pos_strand: buf[j].1 }); } }
            for j in 0..buf_pos { if min.0 == buf[j].0 && buf[j].1 != min.1 { out.push(Minimizer{ key_span: buf[j].0, rid_pos_strand: buf[j].1 }); } }
        }
        if info.0 <= min.0 {
            if l >= (w as i32 + k as i32) && min.0 != u64::MAX { out.push(Minimizer{ key_span: min.0, rid_pos_strand: min.1 }); }
            min = info; min_pos = buf_pos;
        } else if buf_pos == min_pos {
            if l >= (w as i32 + k as i32 - 1) && min.0 != u64::MAX { out.push(Minimizer{ key_span: min.0, rid_pos_strand: min.1 }); }
            min.0 = u64::MAX;
            for j in (buf_pos + 1)..w { if min.0 >= buf[j].0 { min = buf[j]; min_pos = j; } }
            for j in 0..=buf_pos { if min.0 >= buf[j].0 { min = buf[j]; min_pos = j; } }
            if l >= (w as i32 + k as i32 - 1) && min.0 != u64::MAX {
                for j in (buf_pos + 1)..w { if min.0 == buf[j].0 && min.1 != buf[j].1 { out.push(Minimizer{ key_span: buf[j].0, rid_pos_strand: buf[j].1 }); } }
                for j in 0..=buf_pos { if min.0 == buf[j].0 && min.1 != buf[j].1 { out.push(Minimizer{ key_span: buf[j].0, rid_pos_strand: buf[j].1 }); } }
            }
        }
        buf_pos += 1; if buf_pos == w { buf_pos = 0; }
    }
    if min.0 != u64::MAX { out.push(Minimizer{ key_span: min.0, rid_pos_strand: min.1 }); }
}
