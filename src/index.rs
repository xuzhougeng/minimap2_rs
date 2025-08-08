use crate::sketch::{sketch_sequence, Minimizer};
use crate::nt4::nt4;
use noodles_fasta::io::Reader as FastaReader;
use std::fs::File;
use std::io::BufReader;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Write, Read};

#[inline]
fn kroundup64(mut x: usize) -> usize { x-=1; x|=x>>1; x|=x>>2; x|=x>>4; x|=x>>8; x|=x>>16; x|=x>>32; x+1 }

#[inline]
fn mm_seq4_set(S: &mut [u32], o: u64, c: u8) {
    let i = (o >> 3) as usize; // every u32 packs 8 bases
    let shift = ((o & 7) << 2) as usize;
    let v = S[i];
    S[i] = (v & !(0xFu32 << shift)) | (((c as u32) & 0xF) << shift);
}

#[inline]
fn mm_seq4_get(S: &[u32], o: u64) -> u8 {
    let i = (o >> 3) as usize;
    let shift = ((o & 7) << 2) as usize;
    ((S[i] >> shift) & 0xF) as u8
}

#[derive(Clone)]
pub struct IndexSeq { pub name: Option<String>, pub offset: u64, pub len: u32, pub is_alt: bool }

struct Bucket { a: Vec<Minimizer>, p: Vec<u64>, h: Option<HashMap<u64,u64>> }

pub struct Index {
    pub w: i32,
    pub k: i32,
    pub b: i32,
    pub flag: i32,
    pub n_seq: u32,
    pub seq: Vec<IndexSeq>,
    pub S: Vec<u32>,
    B: Vec<Bucket>,
}

pub enum Occurrences<'a> { Single(u64), Multi(&'a [u64]) }

impl Index {
    pub fn new(w: i32, k: i32, b: i32, flag: i32) -> Self {
        let mut B = Vec::with_capacity(1usize << b);
        for _ in 0..(1usize<<b) { B.push(Bucket{ a:Vec::new(), p:Vec::new(), h:None }); }
        Self{ w, k, b, flag, n_seq: 0, seq: Vec::new(), S: Vec::new(), B }
    }

    pub fn get_ref_subseq(&self, rid: usize, st: i32, en: i32) -> Vec<u8> {
        let mut out = Vec::new();
        if rid >= self.seq.len() { return out; }
        let s = &self.seq[rid];
        let mut st0 = st.max(0) as u64;
        let mut en0 = en.min(s.len as i32).max(0) as u64;
        if st0 >= en0 { return out; }
        st0 += s.offset; en0 += s.offset;
        for o in st0..en0 {
            let c = mm_seq4_get(&self.S, o);
            let b = match c { 0=>b'A', 1=>b'C', 2=>b'G', 3=>b'T', _=>b'N' };
            out.push(b);
        }
        out
    }

    fn add_minimizers(&mut self, v: &[Minimizer]) {
        let mask = (1u64 << self.b) - 1;
        for m in v { let idx = ((m.key_span >> 8) & mask as u64) as usize; self.B[idx].a.push(*m); }
    }

    fn post_process(&mut self) {
        let b_bits = self.b;
        // parallel over buckets
        self.B.par_iter_mut().for_each(|b| {
            if b.a.is_empty() { return; }
            b.a.sort_by_key(|x| x.key_span >> 8);
            // count
            let mut n: i32 = 1; let mut n_keys: i32 = 0; let mut total_p: usize = 0;
            for j in 1..=b.a.len() {
                if j==b.a.len() || (b.a[j].key_span>>8)!=(b.a[j-1].key_span>>8) { n_keys+=1; if n>1 { total_p += n as usize; } n=1; } else { n+=1; }
            }
            b.p = vec![0u64; total_p];
            let mut h: HashMap<u64,u64> = HashMap::with_capacity(n_keys as usize);
            // fill
            n=1; let mut start_a = 0usize; let mut start_p = 0usize;
            for j in 1..=b.a.len() {
                if j==b.a.len() || (b.a[j].key_span>>8)!=(b.a[j-1].key_span>>8) {
                    let p = b.a[j-1];
                    let key_top = ((p.key_span>>8) >> b_bits) << 1; // minier>>b << 1
                    if n==1 {
                        h.insert(key_top | 1, p.rid_pos_strand);
                    } else {
                        for k in 0..n { b.p[start_p + k as usize] = b.a[start_a + k as usize].rid_pos_strand; }
                        // sort positions by y
                        b.p[start_p..start_p + n as usize].sort_unstable();
                        let val = ((start_p as u64) << 32) | (n as u64);
                        h.insert(key_top, val);
                        start_p += n as usize;
                    }
                    start_a = j; n=1;
                } else { n+=1; }
            }
            b.h = Some(h);
            b.a.clear();
        });
    }

    pub fn stats(&self) -> (u64, f64, f64, u64) {
        // distinct minimizers n, avg occurrences, avg spacing, total length
        let mut n_keys: u64 = 0;
        let mut sum_occ: u64 = 0;
        for b in &self.B {
            if let Some(ref h) = b.h { for (k,v) in h.iter() { if (k & 1) == 1 { n_keys += 1; sum_occ += 1; } else { n_keys += 1; sum_occ += (*v & 0xffffffff) as u64; } } }
        }
        let mut total_len: u64 = 0; for s in &self.seq { total_len += s.len as u64; }
        let avg_occ = if n_keys>0 { (sum_occ as f64)/(n_keys as f64) } else { 0.0 };
        let avg_spacing = if sum_occ>0 { (total_len as f64)/(sum_occ as f64) } else { 0.0 };
        (n_keys, avg_occ, avg_spacing, total_len)
    }

    pub fn calc_mid_occ(&self, frac: f32) -> i32 {
        // similar to mm_idx_cal_max_occ: find threshold so that (1-frac) quantile of occurrences
        let mut counts: Vec<u32> = Vec::new();
        for b in &self.B {
            if let Some(ref h) = b.h {
                for (k, v) in h.iter() {
                    let c = if (k & 1) == 1 { 1 } else { (*v & 0xffffffff) as u32 };
                    counts.push(c);
                }
            }
        }
        if counts.is_empty() { return i32::MAX; }
        counts.sort_unstable();
        let n = counts.len();
        let idx = ((1.0 - frac as f64) * (n as f64)) as usize;
        let idx = idx.min(n-1);
        (counts[idx] as i32) + 1
    }

    pub fn get<'a>(&'a self, minier: u64) -> Option<Occurrences<'a>> {
        let mask = (1u64 << self.b) - 1;
        let b = &self.B[(minier & mask) as usize];
        let h = b.h.as_ref()?;
        let key = ((minier >> self.b) << 1) as u64;
        if let Some(&val) = h.get(&(key | 1)) { return Some(Occurrences::Single(val)); }
        if let Some(&val) = h.get(&key) {
            let off = (val >> 32) as usize; let n = (val & 0xffffffff) as usize;
            return Some(Occurrences::Multi(&b.p[off..off+n]));
        }
        None
    }

    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        use std::io::Write as _;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        // header
        f.write_all(b"MM2RSIDX\0")?; // magic 9 bytes
        f.write_all(&1u32.to_le_bytes())?; // version
        // basic params
        f.write_all(&self.w.to_le_bytes())?;
        f.write_all(&self.k.to_le_bytes())?;
        f.write_all(&self.b.to_le_bytes())?;
        f.write_all(&self.flag.to_le_bytes())?;
        f.write_all(&self.n_seq.to_le_bytes())?;
        // sequences
        let n_seq = self.seq.len() as u32;
        f.write_all(&n_seq.to_le_bytes())?;
        for s in &self.seq {
            let has_name: u8 = if s.name.is_some() {1} else {0};
            f.write_all(&has_name.to_le_bytes())?;
            if let Some(ref name) = s.name {
                let len = name.len() as u32; f.write_all(&len.to_le_bytes())?; f.write_all(name.as_bytes())?;
            }
            f.write_all(&s.offset.to_le_bytes())?;
            f.write_all(&s.len.to_le_bytes())?;
            f.write_all(&(if s.is_alt {1u8} else {0u8}).to_le_bytes())?;
        }
        // packed sequence array
        let s_len = self.S.len() as u64; f.write_all(&s_len.to_le_bytes())?;
        // write S in large buffered chunks
        const CHUNK_WORDS: usize = 1 << 16; // 65,536 u32 (~256 KiB of bytes per chunk)
        let mut i = 0usize;
        let s_len_words = self.S.len();
        let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_WORDS * 4);
        while i < s_len_words {
            let end = (i + CHUNK_WORDS).min(s_len_words);
            buf.clear();
            for &w in &self.S[i..end] { buf.extend_from_slice(&w.to_le_bytes()); }
            f.write_all(&buf)?;
            i = end;
        }
        // buckets
        let nb = self.B.len() as u32; f.write_all(&nb.to_le_bytes())?;
        for b in &self.B {
            // positions array p
            let p_len = b.p.len() as u64; f.write_all(&p_len.to_le_bytes())?;
            // write p in chunks
            let mut i = 0usize;
            let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_WORDS * 8);
            while i < b.p.len() {
                let end = (i + CHUNK_WORDS).min(b.p.len());
                buf.clear();
                for &v in &b.p[i..end] { buf.extend_from_slice(&v.to_le_bytes()); }
                f.write_all(&buf)?;
                i = end;
            }
            // hash map h
            let has_h: u8 = if b.h.is_some() {1} else {0}; f.write_all(&has_h.to_le_bytes())?;
            if let Some(ref h) = b.h {
                let h_len = h.len() as u64; f.write_all(&h_len.to_le_bytes())?;
                // write hashmap entries buffered
                let mut i = 0usize;
                let mut buf: Vec<u8> = Vec::with_capacity(2 * CHUNK_WORDS * 8);
                for (k, v) in h.iter() {
                    let _ = i; // silence unused in some branches
                    buf.extend_from_slice(&k.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                    if buf.len() >= 2 * CHUNK_WORDS * 8 {
                        f.write_all(&buf)?; buf.clear();
                    }
                }
                if !buf.is_empty() { f.write_all(&buf)?; }
            }
        }
        f.flush()?;
        Ok(())
    }

    // Write index in minimap2 MMI format (compatible with C mm_idx_dump/mm_idx_load)
    pub fn save_to_mmi(&self, path: &str) -> anyhow::Result<()> {
        use std::io::Write as _;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        // header: magic + 5 x u32 (w,k,b,n_seq,flag)
        f.write_all(b"MMI\x02")?;
        let w = self.w as u32;
        let k = self.k as u32;
        let b = self.b as u32;
        let n_seq = self.seq.len() as u32;
        let flag = self.flag as u32;
        for v in [w, k, b, n_seq, flag] { f.write_all(&v.to_le_bytes())?; }
        // sequences: name length (u8), name bytes, len (u32). Offset is implied.
        let mut sum_len: u64 = 0;
        for s in &self.seq {
            if let Some(ref name) = s.name {
                let l = name.len().min(255) as u8; // cap at 255 like C's uint8_t
                f.write_all(&[l])?;
                f.write_all(&name.as_bytes()[..l as usize])?;
            } else {
                f.write_all(&[0u8])?;
            }
            f.write_all(&s.len.to_le_bytes())?;
            sum_len += s.len as u64;
        }
        // buckets: for i in 0..(1<<b): write n (u32), p (u64[n]), size (u32), then size entries of (key,u64,val,u64)
        let nb: usize = 1usize << self.b;
        // Write in bucket-index order to match C
        for i in 0..nb {
            let bkt = &self.B[i];
            let n = bkt.p.len() as u32;
            f.write_all(&n.to_le_bytes())?;
            // write p array buffered
            if n > 0 {
                const CHUNK: usize = 1 << 16;
                let mut off = 0usize;
                let mut buf: Vec<u8> = Vec::with_capacity(CHUNK * 8);
                while off < bkt.p.len() {
                    let end = (off + CHUNK).min(bkt.p.len());
                    buf.clear();
                    for &v in &bkt.p[off..end] { buf.extend_from_slice(&v.to_le_bytes()); }
                    f.write_all(&buf)?;
                    off = end;
                }
            }
            let size = bkt.h.as_ref().map(|h| h.len() as u32).unwrap_or(0u32);
            f.write_all(&size.to_le_bytes())?;
            if let Some(ref h) = bkt.h {
                // write each (key,val)
                let mut buf: Vec<u8> = Vec::with_capacity(2 * 1024);
                for (k, v) in h.iter() {
                    buf.extend_from_slice(&k.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                    if buf.len() >= 2 * 1024 { f.write_all(&buf)?; buf.clear(); }
                }
                if !buf.is_empty() { f.write_all(&buf)?; }
            }
        }
        // packed sequence array at the end, unless NO_SEQ
        if (self.flag & 0) == 0 { // we always include sequences; C checks NO_SEQ bit itself
            let words = ((sum_len + 7) / 8) as usize;
            // write exactly words entries from S (S may be slightly larger due to rounding)
            const CHUNK: usize = 1 << 16;
            let mut off = 0usize;
            let mut buf: Vec<u8> = Vec::with_capacity(CHUNK * 4);
            while off < words {
                let end = (off + CHUNK).min(words);
                buf.clear();
                for &w in &self.S[off..end] { buf.extend_from_slice(&w.to_le_bytes()); }
                f.write_all(&buf)?;
                off = end;
            }
        }
        f.flush()?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> anyhow::Result<Index> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 9]; f.read_exact(&mut magic)?;
        if &magic != b"MM2RSIDX\0" { anyhow::bail!("invalid index file magic"); }
        let mut u32buf = [0u8; 4];
        f.read_exact(&mut u32buf)?; let _ver = u32::from_le_bytes(u32buf);
        // params
        let mut i32buf = [0u8; 4];
        f.read_exact(&mut i32buf)?; let w = i32::from_le_bytes(i32buf);
        f.read_exact(&mut i32buf)?; let k = i32::from_le_bytes(i32buf);
        f.read_exact(&mut i32buf)?; let b = i32::from_le_bytes(i32buf);
        f.read_exact(&mut i32buf)?; let flag = i32::from_le_bytes(i32buf);
        f.read_exact(&mut u32buf)?; let n_seq_decl = u32::from_le_bytes(u32buf);
        // sequences
        f.read_exact(&mut u32buf)?; let n_seq = u32::from_le_bytes(u32buf) as usize;
        let mut seq = Vec::with_capacity(n_seq);
        for _ in 0..n_seq {
            let mut u8buf = [0u8;1]; f.read_exact(&mut u8buf)?; let has_name = u8buf[0] != 0;
            let name = if has_name {
                f.read_exact(&mut u32buf)?; let l = u32::from_le_bytes(u32buf) as usize;
                let mut nb = vec![0u8; l]; f.read_exact(&mut nb)?; Some(String::from_utf8(nb).unwrap_or_else(|_| String::from("*")))
            } else { None };
            let mut u64buf = [0u8;8]; f.read_exact(&mut u64buf)?; let offset = u64::from_le_bytes(u64buf);
            f.read_exact(&mut u32buf)?; let len = u32::from_le_bytes(u32buf);
            let mut u8b = [0u8;1]; f.read_exact(&mut u8b)?; let is_alt = u8b[0] != 0;
            seq.push(IndexSeq{ name, offset, len, is_alt });
        }
        // packed sequence array
        let mut u64buf = [0u8;8]; f.read_exact(&mut u64buf)?; let s_len_words = u64::from_le_bytes(u64buf) as usize;
        let mut S = vec![0u32; s_len_words];
        for i in 0..s_len_words { let mut wbuf=[0u8;4]; f.read_exact(&mut wbuf)?; S[i] = u32::from_le_bytes(wbuf); }
        // buckets
        f.read_exact(&mut u32buf)?; let nb = u32::from_le_bytes(u32buf) as usize;
        let mut B: Vec<Bucket> = Vec::with_capacity(nb);
        for _ in 0..nb {
            f.read_exact(&mut u64buf)?; let p_len = u64::from_le_bytes(u64buf) as usize;
            let mut p = vec![0u64; p_len];
            for i in 0..p_len { let mut buf=[0u8;8]; f.read_exact(&mut buf)?; p[i] = u64::from_le_bytes(buf); }
            let mut u8b=[0u8;1]; f.read_exact(&mut u8b)?; let has_h = u8b[0] != 0;
            let mut h: Option<HashMap<u64,u64>> = None;
            if has_h {
                f.read_exact(&mut u64buf)?; let h_len = u64::from_le_bytes(u64buf) as usize;
                let mut map: HashMap<u64,u64> = HashMap::with_capacity(h_len);
                for _ in 0..h_len { let mut kb=[0u8;8]; let mut vb=[0u8;8]; f.read_exact(&mut kb)?; f.read_exact(&mut vb)?; map.insert(u64::from_le_bytes(kb), u64::from_le_bytes(vb)); }
                h = Some(map);
            }
            B.push(Bucket{ a: Vec::new(), p, h });
        }
        Ok(Index{ w,k,b,flag, n_seq: n_seq_decl, seq, S, B })
    }

    /// Load an index written in minimap2's MMI format by `save_to_mmi`
    pub fn load_from_mmi(path: &str) -> anyhow::Result<Index> {
        let mut f = std::fs::File::open(path)?;
        // magic
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"MMI\x02" { anyhow::bail!("invalid MMI magic"); }
        // header: w,k,b,n_seq,flag (u32)
        let mut u32buf = [0u8; 4];
        f.read_exact(&mut u32buf)?; let w = u32::from_le_bytes(u32buf) as i32;
        f.read_exact(&mut u32buf)?; let k = u32::from_le_bytes(u32buf) as i32;
        f.read_exact(&mut u32buf)?; let b = u32::from_le_bytes(u32buf) as i32;
        f.read_exact(&mut u32buf)?; let n_seq = u32::from_le_bytes(u32buf);
        f.read_exact(&mut u32buf)?; let flag = u32::from_le_bytes(u32buf) as i32;

        // sequences
        let mut seq: Vec<IndexSeq> = Vec::with_capacity(n_seq as usize);
        let mut sum_len: u64 = 0;
        for _ in 0..n_seq {
            let mut lbuf = [0u8; 1];
            f.read_exact(&mut lbuf)?; let name_len = lbuf[0] as usize;
            let name = if name_len > 0 {
                let mut nb = vec![0u8; name_len];
                f.read_exact(&mut nb)?;
                Some(String::from_utf8(nb).unwrap_or_else(|_| String::from("*")))
            } else { None };
            let mut len_buf = [0u8; 4];
            f.read_exact(&mut len_buf)?; let len_u32 = u32::from_le_bytes(len_buf);
            let offset = sum_len;
            sum_len += len_u32 as u64;
            seq.push(IndexSeq{ name, offset, len: len_u32, is_alt: false });
        }

        // buckets
        let nbuckets: usize = 1usize << (b as usize);
        let mut B: Vec<Bucket> = Vec::with_capacity(nbuckets);
        for _ in 0..nbuckets {
            // n (u32), then p array (u64[n])
            let mut nbuf = [0u8; 4];
            f.read_exact(&mut nbuf)?; let n = u32::from_le_bytes(nbuf) as usize;
            let mut p = vec![0u64; n];
            for i in 0..n { let mut vbuf = [0u8; 8]; f.read_exact(&mut vbuf)?; p[i] = u64::from_le_bytes(vbuf); }
            // size (u32), then size entries of (key u64, val u64)
            let mut sbuf = [0u8; 4];
            f.read_exact(&mut sbuf)?; let size = u32::from_le_bytes(sbuf) as usize;
            let mut h: Option<HashMap<u64,u64>> = None;
            if size > 0 {
                let mut map: HashMap<u64,u64> = HashMap::with_capacity(size);
                for _ in 0..size {
                    let mut kb = [0u8; 8]; let mut vb = [0u8; 8];
                    f.read_exact(&mut kb)?; f.read_exact(&mut vb)?;
                    map.insert(u64::from_le_bytes(kb), u64::from_le_bytes(vb));
                }
                h = Some(map);
            }
            B.push(Bucket{ a: Vec::new(), p, h });
        }

        // packed sequence array S: words = (sum_len + 7) / 8
        let words = ((sum_len + 7) / 8) as usize;
        let mut S: Vec<u32> = vec![0u32; words];
        for i in 0..words { let mut wbuf = [0u8; 4]; f.read_exact(&mut wbuf)?; S[i] = u32::from_le_bytes(wbuf); }

        Ok(Index{ w, k, b, flag, n_seq, seq, S, B })
    }
}

pub fn build_index_from_fasta(path: &str, w: i32, k: i32, b: i32, flag: i32) -> anyhow::Result<Index> {
    let mut idx = Index::new(w, k, b, flag);
    // Read all records using noodles
    let file = File::open(path)?;
    let mut reader = FastaReader::new(BufReader::new(file));
    let mut records: Vec<(Option<String>, Vec<u8>)> = Vec::new();
    for result in reader.records() {
        let record = result?;
        let name = Some(String::from_utf8(record.name().to_vec()).unwrap_or_else(|_| "*".to_string()));
        let seq = record.sequence().as_ref().to_vec();
        records.push((name, seq));
    }
    idx.n_seq = records.len() as u32;
    // Precompute minimizers per sequence in parallel
    let is_hpc = (flag & 1) != 0;
    let minis_by_seq: Vec<Vec<Minimizer>> = records
        .par_iter()
        .enumerate()
        .map(|(rid, (_name, seq))| {
            let mut a: Vec<Minimizer> = Vec::new();
            if !seq.is_empty() {
                sketch_sequence(seq, w as usize, k as usize, rid as u32, is_hpc, &mut a);
            }
            a
        })
        .collect();
    // Compute total length and allocate packed sequence storage
    let total_len: u64 = records.iter().map(|(_, s)| s.len() as u64).sum();
    let words = kroundup64(((total_len + 7) / 8) as usize);
    idx.S.resize(words, 0);
    // Pack sequences and metadata, and add minimizers
    let mut sum_len: u64 = 0;
    for (rid, (name, seq)) in records.into_iter().enumerate() {
        // pack
        for (j, &ch) in seq.iter().enumerate() {
            let c = nt4(ch);
            let o = sum_len + j as u64;
            mm_seq4_set(&mut idx.S, o, c);
        }
        // meta
        idx.seq.push(IndexSeq{ name, offset: sum_len, len: seq.len() as u32, is_alt: false });
        // add minimizers for this rid
        idx.add_minimizers(&minis_by_seq[rid]);
        sum_len += seq.len() as u64;
    }
    // finalize buckets
    idx.post_process();
    Ok(idx)
}
