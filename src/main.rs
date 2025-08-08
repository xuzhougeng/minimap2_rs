use clap::{Parser, Subcommand};
use mm2rs::index::build_index_from_fasta;
use mm2rs::index::Index;
use mm2rs::seeds::{collect_query_minimizers, filter_query_minimizers, build_anchors_filtered};
use mm2rs::lchain::{chain_dp, chain_dp_all, ChainParams, select_and_filter_chains, merge_adjacent_chains_with_gap, rescue_long_join};
use mm2rs::paf::{paf_from_chain, write_paf, write_paf_many_with_scores};
use std::fs::File;
use noodles_fasta::io::Reader as FastaReader;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(version, about = "mm2rs: Rust rewrite of minimap2 (WIP)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Index {
        fasta: String,
        #[arg(short = 'w', default_value_t = 10)]
        w: i32,
        #[arg(short = 'k', default_value_t = 15)]
        k: i32,
        #[arg(short = 'b', default_value_t = 14)]
        bucket_bits: i32,
        #[arg(short = 'H', long = "hpc", default_value_t = false)]
        hpc: bool,
            #[arg(short = 'd', long = "dump")] 
            dump: Option<String>,
    },
    Anchors {
        ref_fasta: String,
        qry_fasta: String,
        #[arg(short = 'w', default_value_t = 10)]
        w: i32,
        #[arg(short = 'k', default_value_t = 15)]
        k: i32,
        #[arg(short = 'H', long = "hpc", default_value_t = false)]
        hpc: bool,
    },
    Chain {
        ref_fasta: String,
        qry_fasta: String,
        #[arg(short = 'w', default_value_t = 10)]
        w: i32,
        #[arg(short = 'k', default_value_t = 15)]
        k: i32,
        #[arg(short = 'r', default_value_t = 5000)]
        bw: i32,
        #[arg(short = 'H', long = "hpc", default_value_t = false)]
        hpc: bool,
    },
    Align {
        ref_fasta: String,
        qry_fasta: String,
        // Indexing basics
        #[arg(short = 'w', default_value_t = 10)]
        w: i32,
        #[arg(short = 'k', default_value_t = 15)]
        k: i32,
        #[arg(short = 'H', long = "hpc", default_value_t = false)]
        hpc: bool,
        // Mapping
        #[arg(short = 'f', default_value_t = 2e-4f32)]
        frac_top_repetitive: f32, // -f
        #[arg(short = 'g', default_value_t = 5000)]
        max_gap: i32, // -g
        #[arg(short = 'r')]
        r: Option<String>, // -r NUM[,NUM]
        #[arg(short = 'n', default_value_t = 3)]
        min_cnt: i32, // -n
        #[arg(short = 'm', default_value_t = 40)]
        min_chain_score: i32, // -m
        #[arg(short = 'M', long, default_value_t = 0.5)]
        mask_level: f32, // -M
        #[arg(short = 'p', long, default_value_t = 0.8)]
        pri_ratio: f32, // -p
        #[arg(short = 'N', long, default_value_t = 5)]
        best_n: usize, // -N
        #[arg(short = 'x')]
        preset: Option<String>, // -x
        // I/O
        #[arg(short = 'a', default_value_t = false)]
        out_sam: bool, // -a (ignored; PAF only for now)
        #[arg(short = 'o')]
        output: Option<String>, // -o FILE
    },
}

fn read_fasta_first(path: &str) -> anyhow::Result<(String, Vec<u8>)> {
    use std::io::BufReader;
    let mut reader = FastaReader::new(BufReader::new(File::open(path)?));
    if let Some(result) = reader.records().next() {
        let record = result?;
        let name = String::from_utf8(record.name().to_vec()).unwrap_or_else(|_| "*".to_string());
        let seq = record.sequence().as_ref().to_vec();
        Ok((name, seq))
    } else {
        Ok(("*".to_string(), Vec::new()))
    }
}

fn default_chain_params(k: i32) -> ChainParams {
    let chain_gap_scale = 0.8f32; // minimap2 default
    let chn_pen_gap = 0.01f32 * chain_gap_scale * (k as f32); // opt->chain_gap_scale*0.01*k
    ChainParams{
        max_dist_x: 5000,
        max_dist_y: 5000,
        bw: 500,
        max_chain_iter: 5000,
        min_chain_score: 40,
        min_cnt: 3,
        chn_pen_gap,
        chn_pen_skip: 0.0,
        max_chain_skip: 25,
        max_drop: 500, // ~bw
        bw_long: 20000,
        rmq_rescue_size: 1000,
        rmq_rescue_ratio: 0.1,
    }
}

fn apply_preset(preset: &str, w: &mut i32, k: &mut i32, _hpc: &mut bool) {
    match preset {
        // Minimal presets for convenience; extend as needed
        "map-ont" => { *k = 15; *w = 10; },
        "map-hifi" | "lr:hq" => { *k = 19; *w = 10; },
        "sr" => { *k = 21; *w = 11; },
        _ => {},
    }
}

fn load_index_auto(path: &str, w: i32, k: i32, b: i32, flag: i32) -> anyhow::Result<Index> {
    // If given a minimap2 .mmi index, load directly
    if path.ends_with(".mmi") {
        return Index::load_from_mmi(path);
    }
    // Try loading our native index format; fall back to building from FASTA
    match Index::load_from_file(path) {
        Ok(idx) => Ok(idx),
        Err(_) => build_index_from_fasta(path, w, k, b, flag),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Index{ fasta, w, k, bucket_bits, hpc, dump } => {
            let flag = if hpc {1} else {0};
            let idx = build_index_from_fasta(&fasta, w, k, bucket_bits, flag)?;
            let (n_keys, avg_occ, avg_spacing, total_len) = idx.stats();
            println!("kmer size: {}; skip: {}; is_hpc: {}; #seq: {}", k, w, if hpc {1} else {0}, idx.n_seq);
            println!("distinct minimizers: {} (avg occ {:.2}) avg spacing {:.3} total length {}", n_keys, avg_occ, avg_spacing, total_len);
            if let Some(path) = dump.as_ref() {
                if path.ends_with(".mmi") { idx.save_to_mmi(path)?; } else { idx.save_to_file(path)?; }
            }
        }
        Commands::Anchors{ ref_fasta, qry_fasta, w, k, hpc } => {
            let flag = if hpc {1} else {0};
            let idx = load_index_auto(&ref_fasta, w, k, 14, flag)?;
            let (_qname, q) = read_fasta_first(&qry_fasta)?;
            let mut mv = collect_query_minimizers(&q, w as usize, k as usize);
            filter_query_minimizers(&mut mv, 10, 0.01);
            let mut mid_occ = idx.calc_mid_occ(2e-4f32);
            if mid_occ < 10 { mid_occ = 10; }
            let anchors = build_anchors_filtered(&idx, &mv, q.len() as i32, mid_occ);
            println!("anchors: {}", anchors.len());
            for a in anchors.iter().take(10) { println!("x=0x{:016x} y=0x{:016x}", a.x, a.y); }
        }
        Commands::Chain{ ref_fasta, qry_fasta, w, k, bw, hpc } => {
            let flag = if hpc {1} else {0};
            let idx = load_index_auto(&ref_fasta, w, k, 14, flag)?;
            let (_qname, q) = read_fasta_first(&qry_fasta)?;
            let mut mv = collect_query_minimizers(&q, w as usize, k as usize);
            filter_query_minimizers(&mut mv, 10, 0.01);
            let mut mid_occ = idx.calc_mid_occ(2e-4f32);
            if mid_occ < 10 { mid_occ = 10; }
            let anchors = build_anchors_filtered(&idx, &mv, q.len() as i32, mid_occ);
            let mut p = default_chain_params(k); p.bw = bw; // allow override bw
            let chain = chain_dp(&anchors, &p);
            println!("best_chain_len: {}", chain.len());
            if let (Some(&st), Some(&en)) = (chain.first(), chain.last()) {
                println!("start: x=0x{:016x} y=0x{:016x}", anchors[st].x, anchors[st].y);
                println!("end:   x=0x{:016x} y=0x{:016x}", anchors[en].x, anchors[en].y);
            }
        }
        Commands::Align{ ref_fasta, qry_fasta, mut w, mut k, mut hpc, frac_top_repetitive, max_gap, r, min_cnt, min_chain_score, mask_level, pri_ratio, best_n, preset, out_sam: _out_sam, output } => {
            if let Some(px) = preset.as_ref() { apply_preset(px, &mut w, &mut k, &mut hpc); }
            let flag = if hpc {1} else {0};
            let idx = load_index_auto(&ref_fasta, w, k, 14, flag)?;
            let (qname, q) = read_fasta_first(&qry_fasta)?;
            let mut mv = collect_query_minimizers(&q, w as usize, k as usize);
            filter_query_minimizers(&mut mv, 10, 0.01);
            let mut mid_occ = idx.calc_mid_occ(frac_top_repetitive);
            if mid_occ < 10 { mid_occ = 10; }
            let anchors = build_anchors_filtered(&idx, &mv, q.len() as i32, mid_occ);
            let mut p = default_chain_params(k);
            p.max_dist_x = max_gap; p.max_dist_y = max_gap;
            p.min_cnt = min_cnt; p.min_chain_score = min_chain_score;
            if let Some(r_opt) = r.as_ref() {
                if !r_opt.is_empty() {
                    let parts: Vec<&str> = r_opt.split(',').collect();
                    if let Some(bw_str) = parts.get(0) { if let Ok(v) = bw_str.parse::<i32>() { p.bw = v; } }
                    if let Some(bw_long_str) = parts.get(1) { if let Ok(v) = bw_long_str.parse::<i32>() { p.bw_long = v; } }
                }
            }
            let (chains_all, scores_all) = chain_dp_all(&anchors, &p);
            let mut lines: Vec<String> = Vec::new();
            if chains_all.is_empty() {
                let chain = chain_dp(&anchors, &p);
                if let Some(rec) = paf_from_chain(&idx, &anchors, &chain, &qname, &q) { lines.push(write_paf(&rec)); }
            } else {
                let (chains_rescued, scores_rescued) = rescue_long_join(&anchors, &chains_all, &scores_all, &p, q.len() as i32);
                let chains_merged = merge_adjacent_chains_with_gap(&anchors, &chains_rescued, p.max_dist_y, p.max_dist_y);
                let (chains, _scores, _is_pri, s1, s2) = select_and_filter_chains(&anchors, &chains_merged, &scores_rescued, mask_level, pri_ratio, best_n);
                lines.extend(write_paf_many_with_scores(&idx, &anchors, &chains, s1, s2, &qname, &q));
            }
            if let Some(path) = output.as_ref() {
                if path != "-" {
                    let mut f = File::create(path)?;
                    for l in lines { writeln!(f, "{}", l)?; }
                } else {
                    for l in lines { println!("{}", l); }
                }
            } else {
                for l in lines { println!("{}", l); }
            }
        }
    }
    Ok(())
}
