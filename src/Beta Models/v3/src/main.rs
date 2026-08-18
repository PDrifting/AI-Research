use std::fs::{File,OpenOptions};
use std::io::{Read, Write, Result};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;
use std::thread;
use rand::prelude::SliceRandom;
use once_cell::sync::Lazy;
use core::f64::consts::PI;
use statrs::function::gamma;
use statrs::distribution::{Normal, ContinuousCDF};
use std::sync::LazyLock;
use realfft::RealFftPlanner;
use rustfft::{num_complex::Complex, Fft, FftPlanner, FftPlannerAvx, FftPlannerSse};

const MACHEP:    f64 = 1.11022302462515654042E-16;
const MAXLOG:    f64 = 7.09782712893383996732224E2;

const BIG:       f64 = 4.503599627370496e15;
const BIGINV:    f64 = 2.22044604925031308085e-16;

const TWO_SQRT_PI: f64 = 1.128379167095512574;
const ONE_SQRT_PI: f64 = 0.564189583547756287;
const REL_ERROR:   f64 = 1e-12;

// ---------------------------------------------------------------------------
// Cephes word-encoded float constants for igam
// ---------------------------------------------------------------------------

pub const A_U16: [[u16; 4]; 5] = [
    [0x6661, 0x2733, 0x9850, 0x3F4A],
    [0xE943, 0xB580, 0x7FBD, 0xBF43],
    [0x5EBB, 0x20DC, 0x019F, 0x3F4A],
    [0xA5A1, 0x16B0, 0xC16C, 0xBF66],
    [0x554B, 0x5555, 0x5555, 0x3FB5],
];

pub const B_U16: [[u16; 4]; 6] = [
    [0x6761, 0x8ff3, 0x8901, 0xc095],
    [0xb93e, 0x355b, 0xf234, 0xc0e2],
    [0x89e5, 0xf890, 0x3d73, 0xc114],
    [0xdb51, 0xf994, 0xbc82, 0xc131],
    [0xf20b, 0x0219, 0x4589, 0xc13a],
    [0x055e, 0x5418, 0x0c67, 0xc12a],
];

pub const C_U16: [[u16; 4]; 6] = [
    [0x12b2, 0x1cf3, 0xfd0d, 0xc075],
    [0xd757, 0x7b89, 0xaa0d, 0xc0d0],
    [0x4c9b, 0xb974, 0xeb84, 0xc10a],
    [0x0043, 0x7195, 0x6286, 0xc131],
    [0xf34c, 0x892f, 0x5255, 0xc143],
    [0xe14a, 0x6a11, 0xce4b, 0xc13e],
];

pub static A_F64: Lazy<[f64; 5]> = Lazy::new(|| [
    cephes_words_to_f64(A_U16[0]),
    cephes_words_to_f64(A_U16[1]),
    cephes_words_to_f64(A_U16[2]),
    cephes_words_to_f64(A_U16[3]),
    cephes_words_to_f64(A_U16[4]),
]);

pub static B_F64: Lazy<[f64; 6]> = Lazy::new(|| [
    cephes_words_to_f64(B_U16[0]),
    cephes_words_to_f64(B_U16[1]),
    cephes_words_to_f64(B_U16[2]),
    cephes_words_to_f64(B_U16[3]),
    cephes_words_to_f64(B_U16[4]),
    cephes_words_to_f64(B_U16[5]),
]);

pub static C_F64: Lazy<[f64; 6]> = Lazy::new(|| [
    cephes_words_to_f64(C_U16[0]),
    cephes_words_to_f64(C_U16[1]),
    cephes_words_to_f64(C_U16[2]),
    cephes_words_to_f64(C_U16[3]),
    cephes_words_to_f64(C_U16[4]),
    cephes_words_to_f64(C_U16[5]),
]);

// ---------------------------------------------------------------------------
// NIST Internal helpers
// ---------------------------------------------------------------------------

pub static TEMPLATE_9: LazyLock<Vec<&'static [u8]>> = LazyLock::new(|| {
    const VALUES: [u16; 148] = [
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,37,39,41,43,45,47,51,53,55,57,59,61,63,67,69,
        71,75,77,79,83,85,87,91,93,95,101,103,107,109,111,117,119,123,125,127,131,135,139,143,147,
        151,155,159,163,167,171,175,179,183,187,191,199,207,215,223,239,255,256,272,288,296,304,312,
        320,324,328,332,336,340,344,348,352,356,360,364,368,372,376,380,384,386,388,392,394,400,402,
        404,408,410,416,418,420,424,426,428,432,434,436,440,442,444,448,450,452,454,456,458,460,464,
        466,468,470,472,474,476,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510
    ];

    VALUES.iter().map(|&value| {
        let mut bits = [0u8; 9];
        for i in 0..9 {
            bits[8 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &'static [u8]
    }).collect()
});

pub static TEMPLATE_10: LazyLock<Vec<&'static [u8]>> = LazyLock::new(|| {
    const VALUES: [u16; 284] = [
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,67,
        69,71,73,75,77,79,83,85,87,89,91,93,95,101,103,105,107,109,111,115,117,119,121,123,125,127,
        131,133,135,139,141,143,147,149,151,155,157,159,163,167,171,173,175,179,181,183,187,189,191,
        197,199,203,205,207,213,215,219,221,223,229,235,237,239,245,247,251,253,255,259,263,267,271,
        275,279,283,287,291,295,299,303,307,311,315,319,323,327,331,335,339,343,347,351,355,359,367,
        371,375,379,383,391,399,407,415,423,431,439,447,463,479,511,512,544,560,576,584,592,600,608,
        616,624,632,640,644,648,652,656,664,668,672,676,680,684,688,692,696,700,704,708,712,716,720,
        724,728,732,736,740,744,748,752,756,760,764,768,770,772,776,778,784,786,788,794,800,802,804,
        808,810,816,818,820,824,826,832,834,836,840,842,844,848,850,852,856,860,864,866,868,872,874,
        876,880,882,884,888,890,892,896,898,900,902,904,906,908,912,914,916,918,920,922,928,930,932,
        934,936,938,940,944,946,948,950,952,954,956,960,962,964,966,968,970,972,974,976,978,980,982,
		984,986,988,992,994,996,998,1000,1002,1004,1006,1008,1010,1012,1014,1016,1018,1020,1022		
    ];

    VALUES.iter().map(|&value| {
        let mut bits = [0u8; 10];
        for i in 0..10 {
            bits[9 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &'static [u8]
    }).collect()
});

fn pr_overlapping(u: i32, eta: f64) -> f64 {
    if u == 0 {
        (-eta).exp()
    } else {
        let mut sum = 0.0;
        for l in 1..=u {
            let term =
                -eta
                - (u as f64) * (2.0f64).ln()
                + (l as f64) * eta.ln()
                - safe_lgamma("Pr Overlapping 1", (l + 1) as f64)
                + safe_lgamma("Pr Overlapping 2", u as f64)
                - safe_lgamma("Pr Overlapping 3", l as f64)
                - safe_lgamma("Pr Overlapping 4", (u - l + 1) as f64);
            sum += term.exp();
        }
        sum
    }
}

// ================================================================
//  Math Helpers (Pure Rust, No Crates)
//  These appear at the top of the file.
// ================================================================

// ---------------------------------------------------------------
// Normal CDF using Abramowitz-Stegun approximation
// ---------------------------------------------------------------
pub fn normal_cdf(x: f64) -> f64 {
    // constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p  = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * (-x*x).exp();

    0.5 * (1.0 + sign as f64 * y)
}

// ---------------------------------------------------------------
// Chi-square CDF (lower incomplete gamma approximation)
// ---------------------------------------------------------------
pub fn lower_incomplete_gamma(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;

    for n in 1..100 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < 1e-12 { break; }
    }

    sum * x.powf(s) * (-x).exp()
}

pub fn gamma(z: f64) -> f64 {
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        return PI / ((PI * z).sin() * gamma(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = 0.99999999999980993;

    for (i, &p_i) in p.iter().enumerate() {
        x += p_i / (z + (i as f64) + 1.0);
    }

    let t = z + p.len() as f64 - 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

pub fn chi_square_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 0.0; }

    // Using regularized gamma function approximation
    let a = k / 2.0;
    let g = gamma(a);
    let lower = lower_incomplete_gamma(a, x / 2.0);

    lower / g
}

// ---------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}
// ---------------------------------------------------------------------------
// Cephes math primitives
// ---------------------------------------------------------------------------
pub fn cephes_words_to_f64(words: [u16; 4]) -> f64 {
    let bytes: [u8; 8] = [
        (words[3] >> 8) as u8, (words[3] & 0xFF) as u8,
        (words[2] >> 8) as u8, (words[2] & 0xFF) as u8,
        (words[1] >> 8) as u8, (words[1] & 0xFF) as u8,
        (words[0] >> 8) as u8, (words[0] & 0xFF) as u8,
    ];
    f64::from_be_bytes(bytes)
}

pub fn erf(x: f64) -> f64 {
    let xsqr = x * x;
    if x.abs() > 2.2 {
        return 1.0 - erfc(x);
    }
    let mut sum = x;
    let mut term = x;
    let mut j = 1.0_f64;

    // Safety limit: 10,000 iterations max
    for _ in 0..10000 {
        term *= xsqr / j;
        sum -= term / (2.0 * j + 1.0);
        j += 1.0;
        term *= xsqr / j;
        sum += term / (2.0 * j + 1.0);
        j += 1.0;

        // Escape if we lose precision or hit NaN
        if sum.abs() < 1e-14 || sum.is_nan() || term.is_nan() { break; }
        if (term.abs() / sum.abs()) <= REL_ERROR { break; }
    }
    TWO_SQRT_PI * sum
}

pub fn erfc(x: f64) -> f64 {
    // If x is extremely large, erfc(x) is 0.0. 
    // This prevents entering the continued fraction loop at all.
    if x > 20.0 { return 0.0; }
    if x < -20.0 { return 2.0; }

    if x.abs() < 2.2 { return 1.0 - erf(x); }
    if x < 0.0 { return 2.0 - erfc(-x); }

    let mut a = 1.0_f64;
    let mut b = x;
    let mut c = x;
    let mut d = x * x + 0.5;
    let mut n = 1.0_f64;
    let mut q2 = b / d;
    let mut q1;

    for _ in 0..1000 {
        let t = a * n + b * x; a = b; b = t;
        let t2 = c * n + d * x; c = d; d = t2;
        n += 0.5;
        q1 = q2;
        q2 = b / d;

        // TODO: VET THIS POSSIBLE HARDENED PATH
		// protect against denominator collapse when q2 sits extremely close to 0.0
        // let diff = (q1 - q2).abs();
        // if diff <= REL_ERROR || diff / q2.abs() <= REL_ERROR { break; }

        if q2.is_nan() || q2.is_infinite() { return 0.0; }
        if ((q1 - q2).abs() / q2.abs()) <= REL_ERROR { break; }
    }
    
    let result = ONE_SQRT_PI * (-x * x).exp() * q2;
    if result.is_nan() { 0.0 } else { result }
}

pub fn safe_erf(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("erf[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { -1.0 } else { 1.0 };
    }
    erf(x)
}

pub fn safe_erfc(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("erfc[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { 2.0 } else { 0.0 };
    }
    erfc(x)
}

pub fn cephes_igamc(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 { return 1.0; }
    if x < 1.0 || x < a    { return 1.0 - cephes_igam(a, x); }
    let ax_ln = a * x.ln() - x - cephes_lgam(a);
    if ax_ln < -MAXLOG { return 0.0; }
    let ax = ax_ln.exp();
    let mut y   = 1.0 - a;
    let mut z   = x + y + 1.0;
    let mut c   = 0.0_f64;
    let mut pkm2 = 1.0_f64;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans  = pkm1 / qkm1;
    loop {
        c   += 1.0; y += 1.0; z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else { 1.0 };
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if pk.abs() > BIG {
            pkm2 *= BIGINV; pkm1 *= BIGINV;
            qkm2 *= BIGINV; qkm1 *= BIGINV;
        }
        if t <= MACHEP { break; }
    }
    ans * ax
}

pub fn cephes_igam(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 { return 0.0; }
    if x > 1.0 && x > a     { return 1.0 - cephes_igamc(a, x); }
    let ax_ln = a * x.ln() - x - cephes_lgam(a);
    if ax_ln < -MAXLOG { return 0.0; }
    let ax  = ax_ln.exp();
    let mut r   = a;
    let mut c   = 1.0_f64;
    let mut ans = 1.0_f64;
    loop {
        r   += 1.0;
        c   *= x / r;
        ans += c;
        if c / ans <= MACHEP { break; }
    }
    ans * ax / a
}

pub fn cephes_lgam(x: f64) -> f64 {
    gamma::ln_gamma(x)
}

pub fn safe_igamc(label: &str, a: f64, x: f64) -> f64 {
    if !a.is_finite() || !x.is_finite() {
        eprintln!("igamc[{}]: non-finite a={} x={}", label, a, x);
        return 0.0;
    }
    if a <= 0.0 || x < 0.0 {
        eprintln!("igamc[{}]: invalid a={} x={}", label, a, x);
        return 0.0;
    }
    cephes_igamc(a, x)
}

pub fn lgamma_unsafe(x: f64) -> f64 { gamma::ln_gamma(x) }

pub fn safe_lgamma(label: &str, x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        eprintln!("lgamma[{}]: invalid x = {}", label, x);
        return f64::INFINITY;
    }
    let v = gamma::ln_gamma(x);
    if !v.is_finite() {
        eprintln!("lgamma[{}]: non-finite result for x={}", label, x);
        return f64::INFINITY;
    }
    v
}

pub fn normal_cdf_unsafe(x: f64) -> f64 {
    const SQRT2: f64 = 1.414213562373095048801688724209698078569672;
    if x > 0.0 {
        0.5 * (1.0 + safe_erf("normal_cdf_unsafe 1", x / SQRT2))
    } else {
        0.5 * (1.0 - safe_erf("normal_cdf_unsafe 2", -x / SQRT2))
    }
}

pub fn safe_normal_cdf(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("normal_cdf[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    normal_cdf_unsafe(x)
}

pub fn calculate_best_m(n: usize) -> usize {
    if n >= 1_000_000 { 1000 } else { 500 }
}

#[derive(Clone)]
pub struct Matrix32 {
    pub rows: [u32; 32],
}

impl Matrix32 {
    pub fn new() -> Self { Matrix32 { rows: [0u32; 32] } }

    pub fn from_bits(bits: &[u8], bit_index: usize) -> Self {
        let mut m = Matrix32::new();
        for r in 0..32 {
            let mut row_val: u32 = 0;
            for c in 0..32 {
                let idx = bit_index + r * 32 + c;
                let bit = bits[idx] & 1;
                row_val |= (bit as u32) << c;
            }
            m.rows[r] = row_val;
        }
        m
    }

    pub fn rank(&self) -> usize {
        let mut rows = self.rows.clone();
        let mut rank = 0usize;
        for col in (0..32).rev() {
            let mut pivot = None;
            for r in rank..32 {
                if ((rows[r] >> col) & 1) == 1 { pivot = Some(r); break; }
            }
            if let Some(piv_row) = pivot {
                rows.swap(rank, piv_row);
                for r in 0..32 {
                    if r != rank && ((rows[r] >> col) & 1) == 1 {
                        rows[r] ^= rows[rank];
                    }
                }
                rank += 1;
            }
        }
        rank
    }
}

const ANCHORS: &[(f64, f64)] = &[
    (128.0,   6.55),
	(192.0,   6.94),
    (256.0,   7.17),
	(384.0,   7.44),
    (512.0,   7.59),
	(768.0,   7.74),
    (1024.0,  7.81),
	(2048.0,  7.91),
	(4096.0,  7.95),
    (16384.0, 7.989),
	(65536.0, 7.997191),
	(131072.0, 8.0),	
];

pub fn mean_entropy_from_block_size(block_size: usize) -> f64 {
    let b = block_size as f64;

    // clamp outside range
    if b <= ANCHORS[0].0 {
        return ANCHORS[0].1;
    }
    if b >= ANCHORS[ANCHORS.len() - 1].0 {
        return ANCHORS[ANCHORS.len() - 1].1;
    }

    let lb = b.log2();

    // find segment
    for w in ANCHORS.windows(2) {
        let (b0, e0) = w[0];
        let (b1, e1) = w[1];

        let lb0 = b0.log2();
        let lb1 = b1.log2();

        if lb >= lb0 && lb <= lb1 {
            let t = (lb - lb0) / (lb1 - lb0);
            return e0 + t * (e1 - e0);
        }
    }

    // fallback (should never hit)
    ANCHORS[ANCHORS.len() - 1].1
}

#[inline] pub fn sanitize_p(p: f64) -> f64 { if p.is_nan() || p < 0.0 { 0.0 } else { p } }

// -------------------------------------------------------------------------------

const MAX_SYNAPSES: usize = 2;
const MAX_DELAY: u8 = 23;

// -------------------------
// Gates
// -------------------------
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[repr(u8)]
pub enum GateType {
    XOR,
    NAND,
    OR,
    AND,
    NOR,
}

// -------------------------
// Node
// -------------------------
#[derive(Clone, Debug)]
pub struct Node {
    pub state: u8,
    pub gate_type: GateType,

    // 30 temporal synapses per node
    pub source_indices: [u16; MAX_SYNAPSES],
    pub delays: [u8; MAX_SYNAPSES],
    pub delay_masks: [u64; MAX_SYNAPSES],
	pub initial_delay_masks: [u64; MAX_SYNAPSES], // original seeds
}

// -------------------------
// RuntimeIon
// -------------------------
#[derive(Clone, Debug)]
pub struct RuntimeIon {
    pub nodes: Vec<Node>,
    pub outputs: Vec<usize>,
    pub gate_outs: Vec<u8>,
}

impl RuntimeIon {
    #[inline]
    fn init_node_connections(
	    delay0: u8, delay1: u8
    ) -> ([u16; MAX_SYNAPSES], [u8; MAX_SYNAPSES]) {
        let mut indices = [0u16; MAX_SYNAPSES];
        let mut delays = [0u8; MAX_SYNAPSES];
        
        indices[0] = 0;
		indices[1] = 0;
		delays[0] = delay0;
		delays[1] = delay1;

        (indices, delays)
    }

    pub fn new(node_count: usize, state: u8, delay0: u8, delay1: u8, delay_mask0: u64, delay_mask1: u64, gate: GateType) -> Self {
        let mut nodes = Vec::with_capacity(node_count);

        for i in 0..node_count {
            let (source_indices, delays) = Self::init_node_connections(delay0, delay1);          
            let mut delay_masks = [0u64; MAX_SYNAPSES];
			let mut initial_delay_masks = [0u64; MAX_SYNAPSES];
			
			initial_delay_masks[0] = delay_mask0;
            delay_masks[0] = delay_mask0;
			initial_delay_masks[1] = delay_mask1;
            delay_masks[1] = delay_mask1;
						
            nodes.push(Node {
                state,
                gate_type: gate,
                source_indices,
                delays,
                delay_masks,
				initial_delay_masks,
            });
        }

        let mut outputs: Vec<usize> = (0..node_count).collect();
        
        RuntimeIon {
            nodes,
            outputs,
            gate_outs: vec![0u8; node_count],
        }
    }

    pub fn dump_topology_to<W: Write>(&self, mut w: W) -> Result<()> {
        writeln!(w, "=== RuntimeIon Topology Dump ===")?;
        writeln!(w, "nodes: {}", self.nodes.len())?;
        writeln!(w, "outputs (indices): {:?}\n", self.outputs)?;
		
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(w, "--- Node {} ---", i)?;
            writeln!(w, "  gate_type: {:?}", node.gate_type)?;
            writeln!(w, "  state: {}", node.state)?;

            writeln!(w, "  synapses (source_index -> delay):")?;
            for j in 0..MAX_SYNAPSES {
                let src = node.source_indices[j];
                let d   = node.delays[j];
                let id  = node.initial_delay_masks[j];
				writeln!(w, "    [{}] src: {:3}, delay: {:3} 0x{:016X}", j, src, d, id);
            }
        }

        writeln!(w, "=== End Topology Dump ===")?;
        Ok(())
    }

    // ---------- tick ----------
    pub fn tick(&mut self, out: &mut [u8]) {
        let len = self.nodes.len();

        for i in 0..len {            
			let node = &self.nodes[i];

            // use first two synapses as logical inputs
            let a_idx = node.source_indices[0] as usize;
            let b_idx = node.source_indices[1] as usize;

            let a = self.nodes[a_idx].state;
            let b = self.nodes[b_idx].state;
            
            self.gate_outs[i] = match node.gate_type {
                GateType::XOR  => a ^ b,
                GateType::NAND => (a & b) ^ 1,
                GateType::OR   => a | b,
                GateType::AND  => a & b,
                GateType::NOR  => (a | b) ^ 1,
            };

            //self.gate_outs[i] = !self.nodes[a_idx].state;
        }

        for i in 0..len {
            let gate_out = self.gate_outs[i] as u64;
            let node = &mut self.nodes[i];
            let mut flux = 0u64;

            for j in 0..MAX_SYNAPSES {
                node.delay_masks[j] |= gate_out << node.delays[j];
                flux ^= node.delay_masks[j] & 1;
                node.delay_masks[j] >>= 1;
			}

            if (flux & 1) == 1 {
                node.state ^= 1;
            }
        }

        for (i, &idx) in self.outputs.iter().enumerate() {
            out[i] = self.nodes[idx].state;
        }
    }

    pub fn generate_bits(&mut self, bit_count: usize) -> BitByteStream {
        let mut out = vec![0u8; self.outputs.len()];
        let mut bits = Vec::with_capacity(bit_count);

        while bits.len() < bit_count {
            self.tick(&mut out);
            for &b in out.iter() {
                bits.push(b);
                if bits.len() == bit_count {
                    break;
                }
            }
        }

        BitByteStream::new_from_bits(bits)
    }
	
	pub fn generate_bit(&mut self) -> u8 {
        let mut out = vec![0u8; self.outputs.len()];
        self.tick(&mut out);
        out[0]
    }
}

#[derive(Debug, Clone)]
pub struct BitByteStream {
    pub bits: Vec<u8>,
    pub bit_len: usize,

    pub bytes: Vec<u8>,
    pub byte_len: usize,
    
    pub byte_histogram: [usize; 256],
    pub byte_expected: f64,
}

impl BitByteStream {
    pub fn new_from_bytes(bytes: Vec<u8>) -> Self {
        let mut bits = Vec::with_capacity(bytes.len() * 8);

        for &b in &bytes {
            for i in (0..8).rev() {
                bits.push((b >> i) & 1);
            }
        }

        Self::initialize(bits, bytes)
    }

    pub fn new_from_bits(bits: Vec<u8>) -> Self {
        let bit_len = bits.len();

        // Convert bits → bytes
        let mut bytes = Vec::with_capacity(bit_len / 8);
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | bit;
            }
            bytes.push(byte);
        }

        Self::initialize(bits, bytes)
    }
     
    fn initialize(bits: Vec<u8>, bytes: Vec<u8>) -> Self {
        let bit_len = bits.len();
        let byte_len = bytes.len();

        // --------------------------------
        // Byte histogram
        // --------------------------------
        let mut byte_hist = [0usize; 256];
        for &b in &bytes {
            byte_hist[b as usize] += 1;
        }

        let expected = byte_len as f64 / 256.0;

        // -----------------------------
        // Return unified stream
        // -----------------------------
        Self {
            bits,
            bit_len,
            bytes,
            byte_len,            
            byte_histogram: byte_hist,
            byte_expected: expected,
        }
    }
}

// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// CALIBRATED TESTS
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------

// 1.
pub fn nist_frequency_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    // 

    let sum: i64 = bits.iter().fold(0i64, |acc, &b| {
        acc + (((b & 1) as i64) * 2 - 1)
    });

    let s_obs = (sum.abs() as f64) / (n as f64).sqrt();
    
    sanitize_p(safe_erfc("frequency test", s_obs / 2.0f64.sqrt()))
}

// 2.
pub fn nist_block_frequency_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    const M: usize = 128;
    let n_blocks = n / M;
    // todo: remove this
	// if n_blocks == 0 { return 0.0; }

    let m_f64 = M as f64;
    let mut sum = 0.0_f64;
    
    for block in bits.chunks_exact(M).take(n_blocks) {
        let mut block_sum = 0usize;

        for &b in block {
            block_sum += (b & 1) as usize;
        }

        let pi = (block_sum as f64) / m_f64;
        let v = pi - 0.5;
        sum += v * v;
    }

    let chi_sq = 4.0 * m_f64 * sum;    
    
	// TODO: this is not sanitizing, may throw weirdness sometimes
	cephes_igamc((n_blocks as f64) / 2.0, chi_sq / 2.0)
}

// 3.
fn cusum_core(z: i64, n: usize) -> f64 {
    if z <= 0 {
        return 0.0;
    }

    let n_i = n as i64;
    let n_f = n as f64;
    let sqrt_n = n_f.sqrt();
    let inv_sqrt_n = 1.0 / sqrt_n;
    let zf = z as f64;

    let scale_factor = std::f64::consts::FRAC_1_SQRT_2 * inv_sqrt_n;

    let mut sum1 = 0.0_f64;
    let lower1 = (-n_i / z + 1) / 4;
    let upper1 = (n_i / z - 1) / 4;

    for k in lower1..=upper1 {
        let kf = k as f64;
        let term1 = (4.0 * kf + 1.0) * zf * scale_factor;
        let term2 = (4.0 * kf - 1.0) * zf * scale_factor;

        sum1 += 0.5 * (1.0 + safe_erf("cusum_phi_1", term1));
        sum1 -= 0.5 * (1.0 + safe_erf("cusum_phi_2", term2));
    }

    let mut sum2 = 0.0_f64;
    let lower2 = (-n_i / z - 3) / 4;
    let upper2 = (n_i / z - 1) / 4;

    for k in lower2..=upper2 {
        let kf = k as f64;
        let term1 = (4.0 * kf + 3.0) * zf * scale_factor;
        let term2 = (4.0 * kf + 1.0) * zf * scale_factor;

        sum2 += 0.5 * (1.0 + safe_erf("cusum_phi_3", term1));
        sum2 -= 0.5 * (1.0 + safe_erf("cusum_phi_4", term2));
    }

    sanitize_p(1.0 - sum1 + sum2)
}

fn cusum_z(bits: &[u8]) -> i64 {
    let mut s: isize   = 0;
    let mut sup: isize = 0;
    let mut inf: isize = 0;
    let mut z: isize   = 0;

    for &b in bits {
        if b == 1 { s += 1 } else { s -= 1 }

        if s > sup { sup += 1 }
        if s < inf { inf -= 1 }

        z = sup.max(-inf);
    }

    z as i64
}

// 3A.
pub fn cusum_forward_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.bit_len;
    let z = cusum_z(&stream.bits);
    cusum_core(z, n)
}

// 3B.

// todo: vet this as possible replacement
/*
pub fn cusum_reverse_test(stream: &BitByteStream) -> f64 {
    let n = stream.bit_len;
    let z_rev = cusum_z_optimized(stream.bits.iter().rev().copied());
    cusum_core(z_rev, n)
}
*/
pub fn cusum_reverse_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.bit_len;

    let mut reversed = stream.bits.clone();
    reversed.reverse();

    let z_rev = cusum_z(&reversed);
    cusum_core(z_rev, n)
}

// 4.
pub fn nist_runs_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    let ones = bits.iter().map(|&b| (b & 1) as usize).sum::<usize>() as f64;
    let pi_obs = ones / n as f64;
    let tau = 2.0 / (n as f64).sqrt();

    // todo: remove this
    // if (pi_obs - 0.5).abs() >= tau {
    //     return 0.0;
    // }

    let transitions: usize = bits
        .windows(2)
        .map(|w| ((w[0] & 1) ^ (w[1] & 1)) as usize)
        .sum();

    let v_obs = 1.0 + (transitions as f64);

    let num = v_obs - 2.0 * (n as f64) * pi_obs * (1.0 - pi_obs);
    let den = 2.0 * pi_obs * (1.0 - pi_obs) * (2.0 * n as f64).sqrt();
    
    sanitize_p(erfc((num / den).abs()))
}

// 5.
pub fn nist_longest_run_of_ones_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;
    
    const K: usize = 6;
    const M: usize = 10_000;
    const V_MIN: usize = 10;
    const V_MAX: usize = 16;
    
    const PI: [f64; 7] = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727];

    let n_blocks = n / M;  
    let mut nu = [0usize; 7];

    for block in bits.chunks_exact(M).take(n_blocks) {
        let mut max_run = 0usize;
        let mut run = 0usize;

        for &b in block {
            let mask = (b & 1) as usize;
            run = (run + 1) * mask;
            max_run = max_run.max(run);
        }

        let idx = max_run.saturating_sub(V_MIN).min(K);
        nu[idx] += 1;
    }

    let mut chi_sq = 0.0_f64;
    let n_blocks_f = n_blocks as f64;

    for i in 0..=K {
        let expected = n_blocks_f * PI[i];
        if expected > 0.0 {
            let diff = nu[i] as f64 - expected;
            chi_sq += (diff * diff) / expected;
        }
    }

    sanitize_p(safe_igamc("longest_run_of_ones", (K as f64) / 2.0, chi_sq / 2.0))
}

// 6.
pub fn nist_binary_matrix_rank_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;
    
    const MATRIX_BITS: usize = 32 * 32;
    let n_matrices = n / MATRIX_BITS;

    const P32: f64 = 0.2887880950866024;
    const P31: f64 = 0.5775761901732048;
    const P30: f64 = 0.1336357147401928; // 1.0 - (P32 + P31)

    let mut f32c = 0usize;
    let mut f31c = 0usize;

    for block in bits.chunks_exact(MATRIX_BITS).take(n_matrices) {
        let r = Matrix32::from_bits(block, 0).rank();
        f32c += (r == 32) as usize;
        f31c += (r == 31) as usize;
    }

    let f30c = n_matrices - (f32c + f31c);
    let n_f = n_matrices as f64;

    let chi_sq = (f32c as f64 - n_f * P32).powi(2) / (n_f * P32)
               + (f31c as f64 - n_f * P31).powi(2) / (n_f * P31)
               + (f30c as f64 - n_f * P30).powi(2) / (n_f * P30);

    sanitize_p((-chi_sq / 2.0).exp())
}

// 7.
pub fn nist_dft_spectral_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;
    let half = n / 2;

    let mut input: Vec<f64> = Vec::with_capacity(n);
    for &b in bits {
        input.push(if b == 1 { 1.0 } else { -1.0 });
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let rfft = planner.plan_fft_forward(n);

    let mut spectrum = rfft.make_output_vec();
    rfft.process(&mut input, &mut spectrum).unwrap();

    let upper_bound_sqr = 2.995732274 * (n as f64);

    let mut n_l = 0usize;
    for c in spectrum.iter().take(half) {
        let mag_sqr = c.re * c.re + c.im * c.im;
        if mag_sqr < upper_bound_sqr {
            n_l += 1;
        }
    }

    let n_l = n_l as f64;
    let n_o = 0.95 * (half as f64);
    let variance = (n as f64) * 0.011875;
    let d = (n_l - n_o) / variance.sqrt();

    safe_erfc("DFT", d.abs() / std::f64::consts::SQRT_2)
}

// 8A.
pub fn nist_non_overlapping_template_9_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    const M: usize = 9;
    const N_BLOCKS: usize = 8;

    // todo: remove this
    // if n < M * N_BLOCKS {
    //     return 0.0;
    // }

    let block_size = n / N_BLOCKS;

    let lambda = (block_size as f64 - M as f64 + 1.0) / 2f64.powi(M as i32);
    let var_wj = (block_size as f64)
        * (1.0 / 2f64.powi(M as i32)
           - (2.0 * M as f64 - 1.0) / 2f64.powi(2 * M as i32));

    if lambda <= 0.0 || var_wj <= 0.0 {
        return 0.0;
    }

    let sqrt_var = var_wj.sqrt();

    let blocks: Vec<&[u8]> = bits.chunks_exact(block_size).take(N_BLOCKS).collect();
    let mut wj = [0usize; N_BLOCKS];

    let mut last_p_value = 0.0_f64;

    for sequence in TEMPLATE_9.iter() {
        for (i_idx, block) in blocks.iter().enumerate() {
            let mut w_obs = 0usize;
            let mut j = 0usize;

            while j + M <= block_size {
                if block[j]     == sequence[0] &&
                   block[j + 1] == sequence[1] &&
                   block[j + 2] == sequence[2] &&
                   block[j + 3] == sequence[3] &&
                   block[j + 4] == sequence[4] &&
                   block[j + 5] == sequence[5] &&
                   block[j + 6] == sequence[6] &&
                   block[j + 7] == sequence[7] &&
                   block[j + 8] == sequence[8]
                {
                    w_obs += 1;
                    j += M;
                } else {
                    j += 1;
                }
            }

            wj[i_idx] = w_obs;
        }

        let mut chi_sq = 0.0_f64;
        for i_idx in 0..N_BLOCKS {
            let diff = (wj[i_idx] as f64 - lambda) / sqrt_var;
            chi_sq += diff * diff;
        }

        last_p_value = safe_igamc("non_overlapping_9", (N_BLOCKS as f64) / 2.0, chi_sq / 2.0);
    }

    sanitize_p(last_p_value)
}

// 8B.
pub fn nist_non_overlapping_template_10_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    const M: usize = 10;
    const N_BLOCKS: usize = 8;

    // todo: remove this
    // if n < M * N_BLOCKS {
    //     return 0.0;
    // }

    let block_size = n / N_BLOCKS;

    let lambda = (block_size as f64 - M as f64 + 1.0) / 2f64.powi(M as i32);
    let var_wj = (block_size as f64) * (1.0 / 2f64.powi(M as i32) - (2.0 * M as f64 - 1.0) / 2f64.powi(2 * M as i32));

    if lambda <= 0.0 || var_wj <= 0.0 {
        return 0.0;
    }

    let sqrt_var = var_wj.sqrt();
    let blocks: Vec<&[u8]> = bits.chunks_exact(block_size).take(N_BLOCKS).collect();
    let mut wj = [0usize; N_BLOCKS];

    let mut last_p_value = 0.0_f64;

    for sequence in TEMPLATE_10.iter() {
        for (i_idx, block) in blocks.iter().enumerate() {
            let mut w_obs = 0usize;
            let mut j = 0usize;

            while j + M <= block_size {
                if block[j]     == sequence[0] &&
                   block[j + 1] == sequence[1] &&
                   block[j + 2] == sequence[2] &&
                   block[j + 3] == sequence[3] &&
                   block[j + 4] == sequence[4] &&
                   block[j + 5] == sequence[5] &&
                   block[j + 6] == sequence[6] &&
                   block[j + 7] == sequence[7] &&
                   block[j + 8] == sequence[8] &&
                   block[j + 9] == sequence[9]
                {
                    w_obs += 1;
                    j += M;
                } else {
                    j += 1;
                }
            }

            wj[i_idx] = w_obs;
        }

        let mut chi_sq = 0.0_f64;
        for i_idx in 0..N_BLOCKS {
            let diff = (wj[i_idx] as f64 - lambda) / sqrt_var;
            chi_sq += diff * diff;
        }

        last_p_value = safe_igamc(
            "non_overlapping_10",
            (N_BLOCKS as f64) / 2.0,
            chi_sq / 2.0,
        );
    }

    sanitize_p(last_p_value)
}

// 9.
pub fn nist_overlapping_template_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;
    
    const M: usize = 9;
    const BIG_M: usize = 1032;
    let big_n = n / BIG_M;
    
    // TODO: remove this... 
	// if big_n == 0 { return 0.0; }

    let lambda = (BIG_M - M + 1) as f64 / 512.0; // 2^9 = 512.0
    let eta = lambda / 2.0;
    const K_USIZE: usize = 5;

    let mut nu = [0u32; 6];
    let mut pi = [0.0f64; 6];
    let mut sum_pi = 0.0;

    for i in 0..K_USIZE {
        pi[i] = pr_overlapping(i as i32, eta);
        sum_pi += pi[i];
    }
    pi[K_USIZE] = 1.0 - sum_pi;

    for block in bits.chunks_exact(BIG_M).take(big_n) {
        let mut w_obs = 0usize;

        for j in 0..=(BIG_M - M) {
            if block[j] == 1 &&
               block[j+1] == 1 &&
               block[j+2] == 1 &&
               block[j+3] == 1 &&
               block[j+4] == 1 &&
               block[j+5] == 1 &&
               block[j+6] == 1 &&
               block[j+7] == 1 &&
               block[j+8] == 1 
            {
                w_obs += 1;
            }
        }

        let idx = w_obs.min(K_USIZE);
        nu[idx] += 1;
    }

    let mut chi2 = 0.0f64;
    let n_f = big_n as f64;
    for i in 0..=K_USIZE {
        let expected = n_f * pi[i];
        if expected > 0.0 {
            let diff = nu[i] as f64 - expected;
            chi2 += (diff * diff) / expected;
        }
    }

    sanitize_p(safe_igamc("overlapping_template", (K_USIZE as f64) / 2.0, chi2 / 2.0))
}

// 10.
pub fn nist_universal_maurer_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    const L: usize = 7;
    const P_LEN: usize = 128;      // 128
    const Q: usize = 1280;      // 1280

    let n_over_l = n / L;
    let k = n_over_l - Q;
    
    const EXPECTED: f64 = 6.1962507;
    const VARIANCE: f64 = 3.125;

    let mut t = [0usize; P_LEN];

    const LN_2_INV: f64 = 1.4426950408889634;
    const SQRT2: f64 = 1.4142135623730951;

    let k_f = k as f64;
    let l_f = L as f64;
    let c = 0.7 - 0.8 / l_f + (4.0 + 32.0 / l_f) * k_f.powf(-3.0 / l_f) / 15.0;
    let sigma = c * (VARIANCE / k_f).sqrt();

    let mut chunks = bits.chunks_exact(L);

    for i in 1..=Q {
        let chunk = chunks.next().unwrap();
        let mut dec = 0usize;
        for &bit in chunk {
            dec = (dec << 1) | (bit & 1) as usize;
        }
        t[dec] = i;
    }

    let mut sum = 0.0;
    for i in (Q + 1)..=(Q + k) {
        let chunk = chunks.next().unwrap();
        let mut dec = 0usize;
        for &bit in chunk {
            dec = (dec << 1) | (bit & 1) as usize;
        }

        let last_seen = t[dec];
        if last_seen > 0 {
            sum += (((i - last_seen) as f64).ln()) * LN_2_INV;
        }
        t[dec] = i;
    }

    let phi = sum / k_f;
    let arg = (phi - EXPECTED).abs() / (SQRT2 * sigma);

    sanitize_p(safe_erfc("Maurer", arg))
}

// 11.
pub fn nist_lempel_ziv_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;

    // todo: remove this
    // if n < 1_000_000 {
    //     return 0.5;
    // }

    let c_n = lempel_ziv_phrase_count(bits);

    const MU: f64 = 69588.0;
    const SIGMA: f64 = 73.237260;

    let v = (c_n as f64 - MU) / SIGMA;

    let cdf_value = normal_cdf(v.abs());
    let p = 2.0 * (1.0 - cdf_value);

    sanitize_p(p)
}

fn lempel_ziv_phrase_count(bits: &[u8]) -> usize {
    let n = bits.len();
    let mut dict: HashSet<Vec<u8>> = HashSet::new();
    let mut i = 0usize;
    let mut phrases = 0usize;

    while i < n {
        let mut w = Vec::new();

        loop {
            if i >= n {
                phrases += 1;
                break;
            }

            w.push(bits[i] & 1);
            i += 1;

            if !dict.contains(&w) {
                dict.insert(w);
                phrases += 1;
                break;
            }
        }
    }

    phrases
}

// 12.
pub fn nist_linear_complexity_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = calculate_best_m(n);

    const K: usize = 6;
    let n_blocks = n / m;
    if n_blocks == 0 {
        return 0.0;
    }

    const PI: [f64; 7] = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833];
    let mut nu = [0.0_f64; 7];

    // Workspace buffers
    let mut c = vec![0u8; m];
    let mut b = vec![0u8; m];
    let mut tmp = vec![0u8; m];
    let mut pp = vec![0u8; m];

    // Precompute constants
    let parity1 = (m + 1) % 2;
    let sign1 = if parity1 == 0 { -1.0 } else { 1.0 };
    let mean = (m as f64) / 2.0
        + (9.0 + sign1) / 36.0
        - (1.0 / 2f64.powi(m as i32)) * ((m as f64) / 3.0 + 2.0 / 9.0);

    let parity2 = m % 2;
    let sign2 = if parity2 == 0 { 1.0 } else { -1.0 };

    for block_slice in bits.chunks_exact(m).take(n_blocks) {
        // Reset workspace
        c.fill(0);
        b.fill(0);
        c[0] = 1;
        b[0] = 1;

        let mut l = 0usize;
        let mut m_idx: isize = -1;

        // Berlekamp–Massey
        for n_idx in 0..m {
            let mut d = block_slice[n_idx];

            for i in 1..=l {
                d ^= c[i] & block_slice[n_idx - i];
            }

            if d == 1 {
                tmp.copy_from_slice(&c);
                pp.fill(0);

                // *** C-faithful shift loop ***
                let shift = (n_idx as isize - m_idx) as usize;

                for j in 0..m {
                    if b[j] == 1 {
                        let idx = j + shift;
                        if idx < m {
                            pp[idx] = 1;
                        }
                        // If idx >= m, C would write out-of-bounds (UB),
                        // but algorithmic invariants guarantee it shouldn't happen.
                        // We safely ignore it.
                    }
                }

                // XOR update
                for i in 0..m {
                    c[i] ^= pp[i];
                }

                if l <= n_idx / 2 {
                    l = n_idx + 1 - l;
                    m_idx = n_idx as isize;
                    b.copy_from_slice(&tmp);
                }
            }
        }

        // Compute T_ and bin index
        let t_val = sign2 * ((l as f64) - mean) + 2.0 / 9.0;

        let idx = if t_val <= -2.5 { 0 }
        else if t_val <= -1.5 { 1 }
        else if t_val <= -0.5 { 2 }
        else if t_val <= 0.5 { 3 }
        else if t_val <= 1.5 { 4 }
        else if t_val <= 2.5 { 5 }
        else { 6 };

        nu[idx] += 1.0;
    }

    // Chi-square
    let mut chi_sq = 0.0_f64;
    for i in 0..=K {
        let expected = (n_blocks as f64) * PI[i];
        if expected > 0.0 {
            chi_sq += (nu[i] - expected).powi(2) / expected;
        }
    }

    sanitize_p(safe_igamc("linear_complexity", (K as f64) / 2.0, chi_sq / 2.0))
}

// 13.
#[inline(always)]
fn psi2_optimized(m: usize, n: usize, eps: &[u8], scratch: &mut [u32]) -> f64 {
    if m == 0 || n == 0 {
        return 0.0;
    }

    let num_blocks = n as f64;
    let pow_len = (1usize << (m + 1)) - 1;

    debug_assert!(scratch.len() >= pow_len);
    scratch[..pow_len].fill(0);

    let safe_limit = n.saturating_sub(m);

    // non-wrapping region
    for i in 0..safe_limit {
        let mut k = 1usize;
        for j in 0..m {
            k = (k << 1) | (eps[i + j] & 1) as usize;
        }
        scratch[k - 1] += 1;
    }

    // wrapping region
    for i in safe_limit..n {
        let mut k = 1usize;
        for j in 0..m {
            let idx = i + j;
            let bit = if idx < n { eps[idx] } else { eps[idx - n] };
            k = (k << 1) | (bit & 1) as usize;
        }
        scratch[k - 1] += 1;
    }

    let start = (1usize << m) - 1;
    let end = (1usize << (m + 1)) - 1;
    let mut sum = 0.0_f64;

    for i in start..end {
        let c = scratch[i] as f64;
        sum += c * c;
    }

    sum * ((1usize << m) as f64) / num_blocks - num_blocks
}

// 13A.
pub fn nist_serial_p1_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = stream.bit_len;
    let m = 2;
    let m_i = m as i32;

    let mut scratch = [0u32; 7];

    let psim0 = psi2_optimized(m, n, bits, &mut scratch);
    let psim1 = psi2_optimized(m - 1, n, bits, &mut scratch);
    let del1 = psim0 - psim1;

    let df = 2f64.powi(m_i - 1) / 2.0;
    sanitize_p(safe_igamc("serial_p1", df, del1 / 2.0))
}

// 13B.
pub fn nist_serial_p2_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = 2;
    let m_i = m as i32;

    let mut scratch = [0u32; 7];

    let psim0 = psi2_optimized(m, n, bits, &mut scratch);
    let psim1 = psi2_optimized(m - 1, n, bits, &mut scratch);
    let psim2 = psi2_optimized(m - 2, n, bits, &mut scratch);

    let del2 = psim0 - 2.0 * psim1 + psim2;
    let df = 2f64.powi(m_i - 2) / 2.0;

    sanitize_p(safe_igamc("serial_p2", df, del2 / 2.0))
}

// 14.
pub fn nist_approximate_entropy_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = 2usize;
    let seq_length = n;
    let mut ap_en_arr = [0.0_f64; 2];
    let mut r = 0usize;

    let wrap_buffer = [
        bits[0],
        if n > 1 { bits[1] } else { 0 },
        if n > 2 { bits[2] } else { 0 },
    ];

    for block_size in m..=m + 1 {
        let num_blocks = seq_length;
        let mut p = [0usize; 16]; 

        let safe_limit = num_blocks.saturating_sub(block_size);
        for i in 0..safe_limit {
            let mut k = 1usize;
            for j in 0..block_size {
                k = (k << 1) | (bits[i + j] & 1) as usize;
            }
            p[k - 1] += 1;
        }

        for i in safe_limit..num_blocks {
            let mut k = 1usize;
            for j in 0..block_size {
                let idx = i + j;
                let bit = if idx < num_blocks {
                    bits[idx]
                } else {
                    wrap_buffer[idx - num_blocks]
                };
                k = (k << 1) | (bit & 1) as usize;
            }
            p[k - 1] += 1;
        }

        let mut sum = 0.0_f64;
        let start_index = (1usize << block_size) - 1;
        let limit = 1usize << block_size;
        
        for idx in start_index..(start_index + limit) {
            if p[idx] > 0 {
                let freq = p[idx] as f64 / num_blocks as f64;
                sum += p[idx] as f64 * freq.ln();
            }
        }
        
        sum /= num_blocks as f64;
        ap_en_arr[r] = sum;
        r += 1;
    }

    let ap_en = ap_en_arr[0] - ap_en_arr[1];
    let chi_sq = 2.0 * (seq_length as f64) * (2.0_f64.ln() - ap_en);
    let df = (1usize << (m - 1)) as f64; // df = 2.0

    sanitize_p(safe_igamc("approximate_entropy", df, chi_sq / 2.0))
}

// supports for 15. and 16.
pub struct ExcursionResult {
    pub min_p: f64,
    pub mean_p: f64,
    pub valid_states: usize,
}

pub fn aggregate_excursion_panel(raw: Vec<Option<f64>>) -> ExcursionResult {
    if raw.is_empty() { return ExcursionResult { min_p: 0.0, mean_p: 0.0, valid_states: 0 }; }

    let mut min_p = f64::INFINITY;
    let mut sum_p = 0.0_f64;
    let mut valid_states = 0usize;

    for p in raw {
        if let Some(v) = p {
            if v < min_p { min_p = v; }
            sum_p += v;
            valid_states += 1;
        }
    }

    if valid_states == 0 { return ExcursionResult { min_p: 0.0, mean_p: 0.0, valid_states: 0 }; }

    let mean_p = sum_p / (valid_states as f64);

    ExcursionResult {
        min_p,
        mean_p,
        valid_states,
    }
}

pub struct ExcursionEligibility {
    pub re_valid: bool,
    pub rev_valid: bool,
    pub j_re: usize,
    pub j_rev: usize,
    pub s_k: Vec<i32>,
    pub cycle: Vec<usize>,
}

pub fn validate_excursion_eligibility_unified(stream: &mut BitByteStream) -> ExcursionEligibility {    
    let bits = &stream.bits;
	let n = stream.bit_len;
    let mut s_k = Vec::with_capacity(n);
    let mut cycle = Vec::new();

    let mut current_sum = 2 * (bits[0] as i32) - 1;
    s_k.push(current_sum);

    let mut j_re = 0usize;
    let mut j_rev = 0usize;

    for i in 1..n {
        current_sum += 2 * (bits[i] as i32) - 1;
        s_k.push(current_sum);

        if current_sum == 0 {
            j_re += 1;
            j_rev += 1;
            cycle.push(i);
        }
    }

    if current_sum != 0 {
        j_rev += 1;
    }
    cycle.push(n - 1);

    let re_valid = j_re >= 500;
    let rev_valid = j_rev >= ((0.005 * (n as f64).sqrt()).max(500.0) as usize);

    ExcursionEligibility {
        re_valid,
        rev_valid,
        j_re,
        j_rev,
        s_k,
        cycle,
    }
}

// 15.
pub fn nist_random_excursions_test(j_f: f64, s_k: &[i32], cycle: &[usize]) -> Vec<Option<f64>> {
    const PI: [[f64; 6]; 5] = [
        [0.0; 6],
        [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125],
        [0.75, 0.0625, 0.046875, 0.03515625, 0.0263671875, 0.0791015625],
        [0.8333333333, 0.02777777778, 0.02314814815, 0.01929012346, 0.01607510288, 0.0803755143],
        [0.875, 0.015625, 0.013671875, 0.01196289063, 0.0104675293, 0.0732727051],
    ];

    const STATE_X: [i32; 8] = [-4, -3, -2, -1, 1, 2, 3, 4];

    // nu[k][i]: k = 0..5 (bin), i = 0..7 (state)
    let mut nu = [[0.0_f64; 8]; 6];
    
    let mut counter = [0usize; 8];
    let mut cycle_idx = 0usize;
    let mut next_boundary = if !cycle.is_empty() { cycle[0] } else { s_k.len().saturating_sub(1) };

    for (i, &val) in s_k.iter().enumerate() {
        if val >= -4 && val <= 4 && val != 0 {
            let offset = if val < 0 { 4 } else { 3 };
            let idx = (val + offset) as usize; // -4..4 → 0..7
            counter[idx] += 1;
        }

        if i == next_boundary {
            for k in 0..8 {
                let c = counter[k];
                let bin = c.min(5); // 0..4 → bin 0..4, >=5 → bin 5
                nu[bin][k] += 1.0;
                counter[k] = 0;
            }

            cycle_idx += 1;
            if cycle_idx < cycle.len() {
                next_boundary = cycle[cycle_idx];
            }
        }
    }

    let mut results = Vec::with_capacity(8);

    for (i, &x_state) in STATE_X.iter().enumerate() {
        let abs_x = x_state.abs() as usize;
        let mut chi_sq = 0.0;

        for k in 0..6 {
            let expected = j_f * PI[abs_x][k];
            if expected > 0.0 {
                let diff = nu[k][i] - expected;
                chi_sq += (diff * diff) / expected;
            }
        }

        let p = safe_igamc("random_excursions", 2.5, chi_sq / 2.0).clamp(0.0, 1.0);
        results.push(Some(p));
    }

    results
}

// 16.
pub fn nist_random_excursions_variant_test(j_f: f64, s_k: &[i32]) -> Vec<Option<f64>> {
    const STATE_X: [i32; 18] = [
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
         1,  2,  3,  4,  5,  6,  7,  8,  9
    ];

    let mut state_counts = [0usize; 18];

    for &val in s_k {
        if val >= -9 && val <= 9 && val != 0 {
            let idx = if val < 0 { (val + 9) as usize }
                      else       { (val + 8) as usize };
            state_counts[idx] += 1;
        }
    }

    let mut results = Vec::with_capacity(18);

    for &x_state in &STATE_X {
        let idx = if x_state < 0 { (x_state + 9) as usize }
                  else           { (x_state + 8) as usize };

        let count = state_counts[idx];
        let numerator = ((count as f64) - j_f).abs();
        let denom = (2.0 * j_f * (4.0 * (x_state.abs() as f64) - 2.0)).sqrt();

        let p = if denom > 0.0 {
            safe_erfc("RE Variant", numerator / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };

        results.push(Some(p));
    }

    results
}

// 17.
pub fn nibble_markov_test(stream: &mut BitByteStream) -> f64 {
    let data = &stream.bytes;
    let mut trans = [0u64; 256];
    let mut row_sum = [0u64; 16];
    let mut col_sum = [0u64; 16];
    
    for w in data.windows(2) {
        let a = (w[0] >> 4) as usize;
        let b = (w[1] >> 4) as usize;
        trans[(a << 4) | b] += 1;
        row_sum[a] += 1;
        col_sum[b] += 1;
    }

    let total: f64 = row_sum.iter().sum::<u64>() as f64;
    if total == 0.0 { return 0.0; }

    let mut chi2 = 0.0_f64;
    for a in 0..16 {
        let r_f = row_sum[a] as f64;
        if r_f == 0.0 { continue; }
        
        let a_stride = a << 4;
        for b in 0..16 {
            let o = trans[a_stride | b] as f64;
            if o > 0.0 {
                let e = (r_f * (col_sum[b] as f64)) / total;
                if e > 0.0 {
                    let diff = o - e;
                    chi2 += (diff * diff) / e;
                }
            }
        }
    }

    sanitize_p(1.0 - chi_square_cdf(chi2, 225.0))
}

// 18.
fn byte_entropy(data: &[u8]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let n_f = n as f64;
    let mut h = 0.0;

    for &c in counts.iter() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n_f;
        h -= p * p.log2();
    }

    h
}

pub fn entropy_global_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;

    let scales = [n / 8, n / 4, n / 2, n];
    let mut min_entropy = f64::INFINITY;

    for &len in &scales {
        if len >= 256 {
            let h = byte_entropy(&bytes[..len]);
            if h < min_entropy {
                min_entropy = h;
            }
        }
    }

    if min_entropy.is_infinite() {
        0.0
    } else {
        min_entropy
    }
}

// 19.
pub fn random_walk_radius_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;

    let mut x = 0i64;
    let mut y = 0i64;
    let mut z = 0i64;
    let mut steps = 0usize;

    for chunk in bits.chunks_exact(3) {
        let dx = ((chunk[0] & 1) as i64) * 2 - 1;
        let dy = ((chunk[1] & 1) as i64) * 2 - 1;
        let dz = ((chunk[2] & 1) as i64) * 2 - 1;

        x += dx;
        y += dy;
        z += dz;
        steps += 1;
    }

    if steps < 10 {
        return 0.5;
    }

    let r2 = (x * x + y * y + z * z) as f64;
    let n = steps as f64;
    let stat = r2 / n;
    let df = 3.0;

    sanitize_p(1.0 - chi_square_cdf(stat, df))
}

// 20.
pub fn gap_test(stream: &mut BitByteStream) -> f64 {
    let mut last_seen = [-1isize; 256];
    const MAX_GAP: usize = 255;
    let mut gaps = [0usize; MAX_GAP + 1];

    for (i, &b) in stream.bytes.iter().enumerate() {
        let idx = b as usize;
        let last = last_seen[idx];
        
        if last >= 0 {
            let gap = i - (last as usize) - 1;
            let g = gap.min(MAX_GAP);
            gaps[g] += 1;
        }
        last_seen[idx] = i as isize;
    }

    let total_gaps: usize = gaps.iter().sum();
    if total_gaps == 0 {
        return 0.0;
    }

    let total_gaps_f = total_gaps as f64;
    let mut expected = [0.0_f64; MAX_GAP + 1];
    
    let p_hit = 1.0 / 256.0;
    let q_miss = 255.0 / 256.0;
    let mut q_pow = 1.0_f64;

    for k in 0..MAX_GAP {
        expected[k] = q_pow * p_hit * total_gaps_f;
        q_pow *= q_miss;
    }
    expected[MAX_GAP] = q_pow * total_gaps_f;

    let mut chi_sq = 0.0_f64;
    for k in 0..=MAX_GAP {
        let e = expected[k];
        if e > 0.0 {
            let diff = (gaps[k] as f64) - e;
            chi_sq += (diff * diff) / e;
        }
    }

    sanitize_p(1.0 - chi_square_cdf(chi_sq, MAX_GAP as f64))
}

// 21.
pub fn voronoi_cell_volume_test_fast(stream: &mut BitByteStream) -> f64 {
	const N_POINTS: usize = 128;
    const N_PROBES: usize = 4096;    

    let bytes = &stream.bytes;

    let mut gens = [(0.0_f64, 0.0_f64); N_POINTS];
    let mut counts = [0usize; N_POINTS];

    let inv_u32_max = 1.0 / (u32::MAX as f64);
    let inv_n_probes = 1.0 / (N_PROBES as f64);
    let inv_n_points = 1.0 / (N_POINTS as f64);

    let mut idx = 0usize;

    for i in 0..N_POINTS {
        let x_raw = u32::from_be_bytes([bytes[idx], bytes[idx+1], bytes[idx+2], bytes[idx+3]]);
        let y_raw = u32::from_be_bytes([bytes[idx+4], bytes[idx+5], bytes[idx+6], bytes[idx+7]]);
        idx += 8;
        
        gens[i] = (
            (x_raw as f64) * inv_u32_max,
            (y_raw as f64) * inv_u32_max,
        );
    }

    for _ in 0..N_PROBES {
        let x_raw = u32::from_be_bytes([bytes[idx], bytes[idx+1], bytes[idx+2], bytes[idx+3]]);
        let y_raw = u32::from_be_bytes([bytes[idx+4], bytes[idx+5], bytes[idx+6], bytes[idx+7]]);
        idx += 8;

        let x = (x_raw as f64) * inv_u32_max;
        let y = (y_raw as f64) * inv_u32_max;

        let mut best = f64::INFINITY;
        let mut best_idx = 0usize;

        for (i, &(gx, gy)) in gens.iter().enumerate() {
            let dx = x - gx;
            let dy = y - gy;
            let d2 = (dx * dx) + (dy * dy);
            
            if d2 < best {
                best = d2;
                best_idx = i;
            }
        }
        counts[best_idx] += 1;
    }

    let mut var = 0.0_f64;
    for &c in counts.iter() {
        let a = (c as f64) * inv_n_probes;
        let d = a - inv_n_points;
        var += d * d;
    }
    
    var *= inv_n_points;
    let std_area = var.sqrt();
    let cv = std_area / inv_n_points;

    const EXPECTED_CV: f64 = 0.53;
    let z = (cv - EXPECTED_CV) / 0.10;
    
    sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())))
}

pub fn voronoi_cv_stat(stream: &BitByteStream) -> f64 {
    const N_POINTS: usize = 128;
    const N_PROBES: usize = 4096;

    let bytes = &stream.bytes;
    let mut gens = [(0.0_f64, 0.0_f64); N_POINTS];
    let mut counts = [0usize; N_POINTS];

    let inv_u32 = 1.0 / (u32::MAX as f64);
    let mut idx = 0usize;

    for i in 0..N_POINTS {
        let x = u32::from_be_bytes([bytes[idx], bytes[idx+1], bytes[idx+2], bytes[idx+3]]) as f64 * inv_u32;
        let y = u32::from_be_bytes([bytes[idx+4], bytes[idx+5], bytes[idx+6], bytes[idx+7]]) as f64 * inv_u32;
        idx += 8;
        gens[i] = (x, y);
    }

    for _ in 0..N_PROBES {
        let x = u32::from_be_bytes([bytes[idx], bytes[idx+1], bytes[idx+2], bytes[idx+3]]) as f64 * inv_u32;
        let y = u32::from_be_bytes([bytes[idx+4], bytes[idx+5], bytes[idx+6], bytes[idx+7]]) as f64 * inv_u32;
        idx += 8;

        let mut best = f64::INFINITY;
        let mut best_idx = 0usize;

        for (i, &(gx, gy)) in gens.iter().enumerate() {
            let dx = x - gx;
            let dy = y - gy;
            let d2 = dx*dx + dy*dy;
            if d2 < best {
                best = d2;
                best_idx = i;
            }
        }
        counts[best_idx] += 1;
    }

    let inv_probes = 1.0 / (N_PROBES as f64);
    let inv_points = 1.0 / (N_POINTS as f64);

    let mut var = 0.0;
    for &c in counts.iter() {
        let a = (c as f64) * inv_probes;
        let d = a - inv_points;
        var += d * d;
    }
    var /= N_POINTS as f64;

    let std_area = var.sqrt();
    std_area / inv_points
}

pub fn voronoi_cell_volume_calibrated(stream: &BitByteStream) -> f64 {
    let cv = voronoi_cv_stat(stream);

    const EXPECTED: f64 = 0.581697798778415;
    const VARIANCE: f64 = 0.0029222700055791307;
    const SQRT2: f64 = 1.4142135623730951;

    let sigma = VARIANCE.sqrt();
    let arg = (cv - EXPECTED).abs() / (SQRT2 * sigma);

    sanitize_p(erfc(arg))
}

// 22. 
fn lz76_complexity(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let n = data.len();
    let mut complexity = 1.0;
    let mut i = 0;
    
    while i < n {
        let mut max_len = 0;
        
        for j in 0..i {
            let mut len = 0;
            while i + len < n && j + len < i && data[j + len] == data[i + len] {
                len += 1;
            }
            if len > max_len {
                max_len = len;
            }
        }
        
        if max_len > 0 {
            i += max_len;
        } else {
            complexity += 1.0;
            i += 1;
        }
    }
    
    complexity
}

pub fn ncd_test(stream: &mut BitByteStream) -> f64 {
    let bytes = &stream.bytes;
    const SEGMENT_SIZE: usize = 8;

    if bytes.len() < SEGMENT_SIZE * 2 {
        return 0.5;
    }

    const MIN_PAIRS: usize = 30;
    const EXPECTED_MEAN: f64 = 0.861712;
    const EXPECTED_STD: f64 = 0.053276;

    let chunks_count = bytes.len() / SEGMENT_SIZE;
    if chunks_count < 2 {
        return 0.5;
    }

    let mut ncd_values = Vec::with_capacity(chunks_count - 1);
    let mut concat_buffer = [0u8; SEGMENT_SIZE * 2];

    for i in 0..(chunks_count - 1) {
        let a = &bytes[i * SEGMENT_SIZE..(i + 1) * SEGMENT_SIZE];
        let b = &bytes[(i + 1) * SEGMENT_SIZE..(i + 2) * SEGMENT_SIZE];

        let c_a = lz76_complexity(a);
        let c_b = lz76_complexity(b);

        if c_a <= 0.0 || c_b <= 0.0 {
            continue;
        }

        concat_buffer[..SEGMENT_SIZE].copy_from_slice(a);
        concat_buffer[SEGMENT_SIZE..].copy_from_slice(b);

        let c_ab = lz76_complexity(&concat_buffer);

        let c_min = c_a.min(c_b);
        let c_max = c_a.max(c_b);
        let ncd = (c_ab - c_min) / c_max;

        if (0.0..=1.0).contains(&ncd) {
            ncd_values.push(ncd);
        }
    }

    let n = ncd_values.len();
    if n < MIN_PAIRS {
        return 0.5;
    }

    let mean_ncd = ncd_values.iter().sum::<f64>() / n as f64;
    let standard_error = EXPECTED_STD / (n as f64).sqrt();
    let z = (mean_ncd - EXPECTED_MEAN) / standard_error;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p = 2.0 * (1.0 - normal.cdf(z.abs()));

    p.clamp(0.0, 1.0)
}

// 23.
pub fn maurer_universal_byte(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 3000 { return 0.0; }

    const Q: usize = 2560;
    let bytes = &stream.bytes;

    let mut last_seen = [0usize; 256];

    for i in 0..Q {
        last_seen[bytes[i] as usize] = i;
    }

    let mut sum_logs = 0.0_f64;
    let mut count = 0.0_f64;

    for i in Q..n {
        let sym = bytes[i] as usize;
        let last = last_seen[sym];

        if last > 0 {
            let dist = i - last;
            sum_logs += (dist as f64).log2();
            count += 1.0;
        }

        last_seen[sym] = i;
    }

    if count == 0.0 { return 0.0; }

    let fn_val = sum_logs / count;

    const EXPECTED: f64 = 7.18366195314182;
    const VARIANCE: f64 = 0.000009521175583521322;
    const SQRT2: f64 = 1.4142135623730951;

    let sigma = VARIANCE.sqrt();
    let arg = (fn_val - EXPECTED).abs() / (SQRT2 * sigma);

    sanitize_p(erfc(arg))
}

// 24.
pub struct QuadPanel {
    pub prime: u64,
    pub word_size: usize,

    pub expected_p: f64,
    pub variance_p: f64,

    pub mean_pos: f64,
    pub mean_neg: f64,
    pub mean_total: f64,
    pub mean_chi2: f64,
}

pub static QUAD_PANELS: [QuadPanel; 15] = [
    QuadPanel { prime: 31397, word_size: 2,
        expected_p: 0.5015573148316973, variance_p: 0.08684431455569558,
        mean_pos: 32755.35142361111, mean_neg: 32777.56371527778,
        mean_total: 65532.915138888886, mean_chi2: 1.032123096257527 },

    QuadPanel { prime: 32749, word_size: 2,
        expected_p: 0.49785519042508725, variance_p: 0.08620053354365514,
        mean_pos: 32765.45934027778, mean_neg: 32767.5496875,
        mean_total: 65533.00902777778, mean_chi2: 1.004189115596702 },

    QuadPanel { prime: 64513, word_size: 2,
        expected_p: 0.48767816411838055, variance_p: 0.08646401462657466,
        mean_pos: 32802.74041666667, mean_neg: 32731.274861111113,
        mean_total: 65534.01527777778, mean_chi2: 1.0930963707860892 },

    QuadPanel { prime: 65521, word_size: 2,
        expected_p: 0.5062835643352063, variance_p: 0.08242952219901316,
        mean_pos: 32774.46125, mean_neg: 32759.596805555557,
        mean_total: 65534.05805555556, mean_chi2: 0.9746831095344951 },

    QuadPanel { prime: 65537, word_size: 2,
        expected_p: 0.4951806007148363, variance_p: 0.08072660359700508,
        mean_pos: 32765.93215277778, mean_neg: 32769.08635416667,
        mean_total: 65535.01850694444, mean_chi2: 0.9853728706352615 },

    QuadPanel { prime: 174763, word_size: 3,
        expected_p: 0.5048279294773828, variance_p: 0.08348870928158578,
        mean_pos: 21846.975520833334, mean_neg: 21842.769201388888,
        mean_total: 43689.744722222225, mean_chi2: 0.9782267509395373 },

    QuadPanel { prime: 999431, word_size: 3,
        expected_p: 0.5013259518548159, variance_p: 0.0821943345118838,
        mean_pos: 21847.07159722222, mean_neg: 21842.887118055554,
        mean_total: 43689.958715277775, mean_chi2: 0.9810301552544803 },

    QuadPanel { prime: 1048583, word_size: 3,
        expected_p: 0.5083917415210454, variance_p: 0.08450183454898715,
        mean_pos: 21846.11909722222, mean_neg: 21843.845729166667,
        mean_total: 43689.96482638889, mean_chi2: 0.9847527880823154 },

    QuadPanel { prime: 1677721, word_size: 3,
        expected_p: 0.5048994361688108, variance_p: 0.08439022665191563,
        mean_pos: 21846.64340277778, mean_neg: 21843.323576388888,
        mean_total: 43689.96697916667, mean_chi2: 0.9980882896014371 },

    QuadPanel { prime: 16777199, word_size: 3,
        expected_p: 0.5101935930073521, variance_p: 0.08536835240122245,
        mean_pos: 21846.756006944444, mean_neg: 21843.239756944444,
        mean_total: 43689.99576388889, mean_chi2: 0.9675851222573184 },

    QuadPanel { prime: 1000000007, word_size: 4,
        expected_p: 0.5016054601701475, variance_p: 0.08306340449077505,
        mean_pos: 16386.19923611111, mean_neg: 16381.800763888888,
        mean_total: 32768.0, mean_chi2: 1.0127510918511284 },

    QuadPanel { prime: 2147483647, word_size: 4,
        expected_p: 0.5064455478262015, variance_p: 0.08248682477252446,
        mean_pos: 16386.8365625, mean_neg: 16381.1634375,
        mean_total: 32768.0, mean_chi2: 0.9676688936021592 },

    QuadPanel { prime: 3221225473, word_size: 4,
        expected_p: 0.5054866339369622, variance_p: 0.08352999762430577,
        mean_pos: 16388.489097222224, mean_neg: 16379.510902777778,
        mean_total: 32768.0, mean_chi2: 0.9882483079698351 },

    QuadPanel { prime: 4294967087, word_size: 4,
        expected_p: 0.4983869152697782, variance_p: 0.08324675266316967,
        mean_pos: 16379.262430555556, mean_neg: 16388.737569444445,
        mean_total: 32768.0, mean_chi2: 1.0290229373508029 },

    QuadPanel { prime: 4294967291, word_size: 4,
        expected_p: 0.5028040730211777, variance_p: 0.08636243845152053,
        mean_pos: 16378.318472222221, mean_neg: 16389.68152777778,
        mean_total: 32768.0, mean_chi2: 1.0200850338406033 },
];

#[inline(always)]
fn legendre_symbol_u64(a: u64, p: u64) -> i64 {
    // Bit-shift replaces division by 2
	let e = (p - 1) >> 1; 
    let r = modexp_u64(a, e, p);
    if      r == 1 {  1 }
	else if r == 0 {  0 }
	else           { -1 }
}

#[inline(always)]
fn modexp_u64(a: u64, mut e: u64, m: u64) -> u64 {
    let mut r: u64 = 1;
    let mut base: u64 = a % m;
    
    while e > 0 {        
        if e & 1 == 1 { r = (r * base) % m; }
        base = (base * base) % m;
        e >>= 1;
    }
    r
}

pub fn quadratic_panel_counts(
    stream: &BitByteStream,
    prime: u64,
    word_size: usize,
) -> (usize, usize, usize) {
    let bytes = &stream.bytes;
    let mut count_pos = 0usize;
    let mut count_neg = 0usize;

    match word_size {
        2 => {
            for chunk in bytes.chunks_exact(2) {
                let w = ((chunk[0] as u64) << 8) | (chunk[1] as u64);
                let a = w % prime;
                if a != 0 {
                    let ls = legendre_symbol_u64(a, prime);
                    count_pos += (ls == 1) as usize;
                    count_neg += (ls == -1) as usize;
                }
            }
        }
        3 => {
            for chunk in bytes.chunks_exact(3) {
                let w = ((chunk[0] as u64) << 16)
                      | ((chunk[1] as u64) << 8)
                      |  (chunk[2] as u64);
                let a = w % prime;
                if a != 0 {
                    let ls = legendre_symbol_u64(a, prime);
                    count_pos += (ls == 1) as usize;
                    count_neg += (ls == -1) as usize;
                }
            }
        }
        4 => {
            for chunk in bytes.chunks_exact(4) {
                let w = ((chunk[0] as u64) << 24)
                      | ((chunk[1] as u64) << 16)
                      | ((chunk[2] as u64) << 8)
                      |  (chunk[3] as u64);
                let a = w % prime;
                if a != 0 {
                    let ls = legendre_symbol_u64(a, prime);
                    count_pos += (ls == 1) as usize;
                    count_neg += (ls == -1) as usize;
                }
            }
        }
        _ => {}
    }

    let total_nonzero = count_pos + count_neg;
    (count_pos, count_neg, total_nonzero)
}

#[inline(always)]
fn quadratic_panel_calibrated(
    stream: &BitByteStream,
    panel: &QuadPanel,
) -> (f64, f64) {
    let (count_pos, count_neg, total_nonzero) =
        quadratic_panel_counts(stream, panel.prime, panel.word_size);

    if total_nonzero == 0 {
        return (1.0, 0.0);
    }

    let expected = (total_nonzero as f64) * 0.5;

    let chi2_raw =
        ((count_pos as f64 - expected).powi(2) / expected)
      + ((count_neg as f64 - expected).powi(2) / expected);

    let p_raw = sanitize_p(1.0 - chi_square_cdf(chi2_raw, 1.0));

    let sigma_p = panel.variance_p.sqrt();
    let z_p = (p_raw - panel.expected_p).abs() / (1.4142135623730951 * sigma_p);
    let p_calibrated = sanitize_p(erfc(z_p));

    let dev_pos = (count_pos as f64 - panel.mean_pos) / panel.mean_total;
    let dev_neg = (count_neg as f64 - panel.mean_neg) / panel.mean_total;
    let dev_balance = dev_pos - dev_neg;

    (p_calibrated, dev_balance)
}

pub fn quadratic_character_multi_panel_metrics(
    stream: &BitByteStream,
) -> (f64, f64, f64, f64) {
    let mut lowest_p = 1.0_f64;
    let mut sum_p = 0.0_f64;
    let mut sum_dev = 0.0_f64;
    let mut max_dev = 0.0_f64;
    let mut count = 0.0_f64;

    for panel in QUAD_PANELS.iter() {
        let (p, dev) = quadratic_panel_calibrated(stream, panel);

        if p < lowest_p {
            lowest_p = p;
        }

        let dev_abs = dev.abs();
        if dev_abs > max_dev {
            max_dev = dev_abs;
        }

        sum_p += p;
        sum_dev += dev;
        count += 1.0;
    }

    let mean_p = if count > 0.0 { sum_p / count } else { 0.0 };
    let mean_dev = if count > 0.0 { sum_dev / count } else { 0.0 };

    (lowest_p, mean_p, max_dev, mean_dev)
}

// 25.
pub struct EscPanel {
    pub block_size: usize,

    pub expected_p: f64,
    pub variance_p: f64,

    pub expected_c: f64,
    pub variance_c: f64,

    pub expected_mean_h: f64,
    pub variance_mean_h: f64,

    pub expected_var_h: f64,
    pub variance_var_h: f64,

    pub expected_min_h: f64,
    pub expected_max_h: f64,
}

pub static ESC_PANELS: [EscPanel; 5] = [
    EscPanel {
        block_size: 256,
        expected_p:      0.5008020129104729,
        variance_p:      0.08453649282264528,
        expected_c:     -0.0000000013755471421923884,
        variance_c:      0.000000000000014273771603129526,
        expected_mean_h: 7.174955045634825,
        variance_mean_h: 0.000005262550105029639,
        expected_var_h:  0.0027473679007930124,
        variance_var_h:  0.000000028213138252511838,
        expected_min_h:  7.007305238961838,
        expected_max_h:  7.328320762759573,
    },
    EscPanel {
        block_size: 384,
        expected_p:      0.5030703033041584,
        variance_p:      0.08241534036926561,
        expected_c:     -0.00000000017989493030019576,
        variance_c:      0.00000000000006776304480437387,
        expected_mean_h: 7.444652713667191,
        variance_mean_h: 0.000004954510487713059,
        expected_var_h:  0.0017252851842387757,
        variance_var_h:  0.000000017112774835363007,
        expected_min_h:  7.318035618388585,
        expected_max_h:  7.561271543337204,
    },
    EscPanel {
        block_size: 512,
        expected_p:      0.4940296645647846,
        variance_p:      0.08355796892655286,
        expected_c:      0.000000020346950895987356,
        variance_c:      0.00000000000019096688214929685,
        expected_mean_h: 7.590402632766508,
        variance_mean_h: 0.000004459157076475904,
        expected_var_h:  0.0011290620482720096,
        variance_var_h:  0.000000009599951387406611,
        expected_min_h:  7.490464813327491,
        expected_max_h:  7.681275359557297,
    },
    EscPanel {
        block_size: 768,
        expected_p:      0.49299515751235956,
        variance_p:      0.08466632597917538,
        expected_c:      0.000000025376569399337294,
        variance_c:      0.0000000000007265216843529444,
        expected_mean_h: 7.737793380862447,
        variance_mean_h: 0.0000030782215831155815,
        expected_var_h:  0.0005384874532191945,
        variance_var_h:  0.00000000339819795236307,
        expected_min_h:  7.671603506921635,
        expected_max_h:  7.796700295635358,
    },
    EscPanel {
        block_size: 1024,
        expected_p:      0.49167999876960183,
        variance_p:      0.08437955354950638,
        expected_c:      0.00000001544931443495183,
        variance_c:      0.0000000000016189893749596107,
        expected_mean_h: 7.80873899639531,
        variance_mean_h: 0.0000022316711567400164,
        expected_var_h:  0.0002941264272865146,
        variance_var_h:  0.0000000013065668862171667,
        expected_min_h:  7.7615417989354345,
        expected_max_h:  7.850454901941898,
    },
];

pub fn entropy_surface_curvature_test_metrics(
    stream: &mut BitByteStream,
    block_size: usize,
    entropies: &mut [f64],
) -> (f64, f64, f64, f64, f64, f64) {

    let bytes = &stream.bytes;
    let n = stream.byte_len;
    if n < block_size * 5 {
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut m = 0usize;
    let mut min_h = f64::INFINITY;
    let mut max_h = 0.0_f64;

    let total = block_size as f64;
    let inv_total = 1.0 / total;

    for blk in bytes.chunks_exact(block_size) {
        let mut freq = [0usize; 256];
        for &b in blk {
            freq[b as usize] += 1;
        }

        let mut h = 0.0_f64;
        for &count in freq.iter() {
            if count > 0 {
                let p = (count as f64) * inv_total;
                h -= p * p.log2();
            }
        }

        if h < min_h { min_h = h; }
        if h > max_h { max_h = h; }

        entropies[m] = h;
        m += 1;
    }

    if m < 5 {
        return (1.0, 0.0, 0.0, 0.0, min_h, max_h);
    }

    let mut s = [0.0_f64; 5];
    let mut t = [0.0_f64; 3];
    let mut sum_h2 = 0.0_f64;

    for idx in 0..m {
        let h = entropies[idx];
        let x = idx as f64;
        let x2 = x * x;

        s[0] += 1.0;
        s[1] += x;
        s[2] += x2;
        s[3] += x2 * x;
        s[4] += x2 * x2;

        t[0] += h;
        t[1] += x * h;
        t[2] += x2 * h;
        sum_h2 += h * h;
    }

    let det = s[0] * (s[2] * s[4] - s[3] * s[3])
            - s[1] * (s[1] * s[4] - s[2] * s[3])
            + s[2] * (s[1] * s[3] - s[2] * s[2]);

    if det.abs() < 1e-12 {
        return (1.0, 0.0, 0.0, 0.0, min_h, max_h);
    }

    let c = (s[0] * (s[2] * t[2] - t[1] * s[3])
           - s[1] * (s[1] * t[2] - t[1] * s[2])
           + t[0] * (s[1] * s[3] - s[2] * s[2])) / det;

    let m_f = m as f64;
    let mean_h = t[0] / m_f;
    let var_h = (sum_h2 / m_f - mean_h * mean_h).max(0.0001);

    let se_c = (var_h * ((s[0] * s[2] - s[1] * s[1]) / det).abs()).sqrt();
    let z = c / se_c;

    let p_curve = 1.0 - erf(z.abs() * std::f64::consts::FRAC_1_SQRT_2);

    let p_final = sanitize_p(p_curve);

    (p_final, c, mean_h, var_h, min_h, max_h)
}

#[inline(always)]
fn entropy_surface_curvature_panel_metrics(
    stream: &mut BitByteStream,
    panel: &EscPanel,
    scratchpad: &mut [f64],
) -> (f64, f64) {
    let (p_raw, c, mean_h, var_h, min_h, max_h) =
        entropy_surface_curvature_test_metrics(stream, panel.block_size, scratchpad);

    // Calibrated p: normalize p_raw against expected_p, variance_p
    let sigma_p = panel.variance_p.sqrt();
    let z_p = if sigma_p > 0.0 {
        (p_raw - panel.expected_p) / (1.4142135623730951 * sigma_p)
    } else {
        0.0
    };
    let p_calibrated = sanitize_p(erfc(z_p.abs()));

    // Curvature + entropy deviation: combine into a single structural deviation metric
    let sigma_c = panel.variance_c.sqrt();
    let z_c = if sigma_c > 0.0 {
        (c - panel.expected_c) / sigma_c
    } else {
        0.0
    };

    let z_mean_h = if panel.variance_mean_h > 0.0 {
        (mean_h - panel.expected_mean_h) / panel.variance_mean_h.sqrt()
    } else {
        0.0
    };

    let z_var_h = if panel.variance_var_h > 0.0 {
        (var_h - panel.expected_var_h) / panel.variance_var_h.sqrt()
    } else {
        0.0
    };

    let z_min_h = (min_h - panel.expected_min_h); // range anchors, no variance needed
    let z_max_h = (max_h - panel.expected_max_h);

    // Aggregate deviation: ESC is insanely sensitive, so keep it simple but expressive
    let dev_struct =
        z_c.abs()
      + z_mean_h.abs()
      + z_var_h.abs()
      + z_min_h.abs()
      + z_max_h.abs();

    (p_calibrated, dev_struct)
}

pub fn entropy_surface_curvature_multi_panel_metrics(
    stream: &mut BitByteStream,
) -> (f64, f64, f64, f64) {
    let mut lowest_p = 1.0_f64;
    let mut sum_p = 0.0_f64;
    let mut sum_dev = 0.0_f64;
    let mut max_dev = 0.0_f64;
    let mut count = 0.0_f64;

    let mut scratchpad = vec![0.0_f64; 400_000];

    for panel in ESC_PANELS.iter() {
        let (p, dev) = entropy_surface_curvature_panel_metrics(stream, panel, &mut scratchpad);

        if p < lowest_p {
            lowest_p = p;
        }

        if dev.abs() > max_dev {
            max_dev = dev.abs();
        }

        sum_p += p;
        sum_dev += dev;
        count += 1.0;
    }

    let mean_p = if count > 0.0 { sum_p / count } else { 0.0 };
    let mean_dev = if count > 0.0 { sum_dev / count } else { 0.0 };

    (lowest_p, mean_p, max_dev, mean_dev)
}

// 26.
// Correct expected distributions (empirical uniform)
pub const EXPECTED_LOCAL_PROBS: [f64; 8] = [
    1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,
    1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,
];

pub const EXPECTED_GLOBAL_PROBS: [f64; 18] = [
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0,
];

pub struct UegPanel {
    pub expected_local_c1: f64,
    pub variance_local_c1: f64,
    pub expected_global_c1: f64,
    pub variance_global_c1: f64,
    pub expected_local_c2: f64,
    pub variance_local_c2: f64,
    pub expected_global_c2: f64,
    pub variance_global_c2: f64,
}

pub const UEG_PANEL: UegPanel = UegPanel {
    expected_local_c1: 0.3481755447471015,
    variance_local_c1: 0.0007738360495205654,

    expected_global_c1: 0.2310010831843693,
    variance_global_c1: 0.000004674837986041919,

    expected_local_c2: 0.3481755447471015,
    variance_local_c2: 0.0007738360495205654,

    expected_global_c2: 0.2295236738926635,
    variance_global_c2: 0.00034001186224734696,
};


fn compute_local_dev_c1(walk: &[i32], total: usize) -> f64 {
    if total == 0 { return 0.0; }

    let mut counts = [0usize; 8];

    for &s in walk {
        if s >= -4 && s <= 4 && s != 0 {
            let idx = if s < 0 { (s + 4) as usize } else { (s + 3) as usize };
            counts[idx] += 1;
        }
    }

    let expected = EXPECTED_LOCAL_PROBS;

    let mut dev = 0.0;
    for i in 0..8 {
        let obs = counts[i] as f64 / (total as f64);
        let diff = obs - expected[i];
        dev += diff * diff;
    }

    dev.sqrt()
}

fn compute_global_dev_c1(walk: &[i32], total: usize) -> f64 {
    if total == 0 { return 0.0; }

    let mut counts = [0usize; 18];

    for &s in walk {
        if s >= -9 && s <= 9 && s != 0 {
            let idx = if s < 0 { (s + 9) as usize } else { (s + 8) as usize };
            counts[idx] += 1;
        }
    }

    let expected = EXPECTED_GLOBAL_PROBS;

    let mut dev = 0.0;
    for i in 0..18 {
        let obs = counts[i] as f64 / (total as f64);
        let diff = obs - expected[i];
        dev += diff * diff;
    }

    dev.sqrt()
}

fn compute_local_dev_c2(walk: &[i32], total: usize) -> f64 {
    if total == 0 { return 0.0; }

    let mut counts = [0usize; 8];

    for &s in walk {
        if s >= -4 && s <= 4 && s != 0 {
            let idx = if s < 0 { (s + 4) as usize } else { (s + 3) as usize };
            counts[idx] += 1;
        }
    }

    let expected = EXPECTED_LOCAL_PROBS;

    let mut dev = 0.0;
    for i in 0..8 {
        let obs = counts[i] as f64 / (total as f64);
        let diff = obs - expected[i];
        dev += diff * diff;
    }

    dev.sqrt()
}

fn compute_global_dev_c2(walk: &[i32], total: usize) -> f64 {
    if total == 0 { return 0.0; }

    let mut counts = [0usize; 18];

    for &s in walk {
        if s >= -9 && s <= 9 && s != 0 {
            let idx = if s < 0 { (s + 9) as usize } else { (s + 8) as usize };
            counts[idx] += 1;
        }
    }

    let expected = EXPECTED_GLOBAL_PROBS;

    let mut dev = 0.0;
    for i in 0..18 {
        let obs = counts[i] as f64 / (total as f64);
        let diff = obs - expected[i];
        dev += diff * diff;
    }

    dev.sqrt()
}

pub struct UnifiedExcursion {
    pub local_c1: f64,
    pub global_c1: f64,
    pub local_c2: f64,
    pub global_c2: f64,
    pub re_valid: bool,
    pub rev_valid: bool,
}

// TODO: there is probably way to combine with the RE/REV test hardness
//       the re_valid, and rev_valid are being checked elsewhere
//       not critical right now.
pub fn unified_excursion_geometry_from_elig(
    s_k: &[i32],
    j_re: usize,
    j_rev: usize,
    re_valid: bool,
    rev_valid: bool,
) -> Option<UnifiedExcursion> {
	
    if !re_valid && !rev_valid {
        return None;
    }

    let local_c1 = if re_valid {
        compute_local_dev_c1(s_k, j_re)
    } else {
        0.0
    };

    let global_c1 = if re_valid {
        compute_global_dev_c1(s_k, j_re)
    } else {
        0.0
    };

    let local_c2 = if rev_valid {
        compute_local_dev_c2(s_k, j_rev)
    } else {
        0.0
    };

    let global_c2 = if rev_valid {
        compute_global_dev_c2(s_k, j_rev)
    } else {
        0.0
    };

    Some(UnifiedExcursion {
        local_c1,
        global_c1,
        local_c2,
        global_c2,
        re_valid,
        rev_valid,
    })
}

#[inline(always)]
fn push_metric(val: f64, mean: f64, var: f64, pvals: &mut Vec<f64>, devs: &mut Vec<f64>) {
    let z = (val - mean) / var.sqrt();
    let p = sanitize_p(erfc(z.abs() / 1.4142135623730951));
    pvals.push(p);
    devs.push(z.abs());
}

pub fn unified_excursion_calibrated(ueg: &UnifiedExcursion) -> Option<(f64, f64, f64, f64)> {
    let mut pvals = Vec::with_capacity(4);
    let mut devs  = Vec::with_capacity(4);

    if ueg.re_valid {
        push_metric(ueg.local_c1,  UEG_PANEL.expected_local_c1,  UEG_PANEL.variance_local_c1,  &mut pvals, &mut devs);
        push_metric(ueg.global_c1, UEG_PANEL.expected_global_c1, UEG_PANEL.variance_global_c1, &mut pvals, &mut devs);
    }

    if ueg.rev_valid {
        push_metric(ueg.local_c2,  UEG_PANEL.expected_local_c2,  UEG_PANEL.variance_local_c2,  &mut pvals, &mut devs);
        push_metric(ueg.global_c2, UEG_PANEL.expected_global_c2, UEG_PANEL.variance_global_c2, &mut pvals, &mut devs);
    }

    
    let lowest_p = pvals.iter().copied().fold(1.0, f64::min);
    let mean_p   = pvals.iter().sum::<f64>() / (pvals.len() as f64);

    let max_dev  = devs.iter().copied().fold(0.0, f64::max);
    let mean_dev = devs.iter().sum::<f64>() / (devs.len() as f64);

    Some((lowest_p, mean_p, max_dev, mean_dev))
}

// 27.
#[derive(Debug, Clone)]
pub struct WalshMetrics {
    pub non_linearity: f64,
    pub spectrum_entropy: f64,
    pub bent_distance: f64,
    pub max_coeff: i32,
    pub plateau_ratio: f64,
    pub unique_magnitudes: usize,
}

/// Computes Walsh-Hadamard spectrum metrics for a Boolean truth table
/// represented by BitByteStream.bits (Vec<u8> of 0/1).
/// Length must be a power of two (2^n).
pub fn walsh_hadamard_metrics(stream: &BitByteStream) -> WalshMetrics {
    let bits = &stream.bits;
    let len = bits.len();

    if len == 0 || !len.is_power_of_two() {
        return WalshMetrics {
            non_linearity: 0.0,
            spectrum_entropy: 0.0,
            bent_distance: 0.0,
            max_coeff: 0,
            plateau_ratio: 0.0,
            unique_magnitudes: 0,
        };
    }

    // Number of variables: len = 2^n_vars
    let n_vars = len.trailing_zeros() as usize;

    // Map {0,1} = {-1,+1}
    let mut w: Vec<i32> = bits.iter()
        .map(|&b| if b == 0 { -1 } else { 1 })
        .collect();

    // Fast Walsh-Hadamard Transform (in-place)
    let mut h = 1;
    while h < len {
        let mut i = 0;
        while i < len {
            for j in i..(i + h) {
                let x = w[j];
                let y = w[j + h];
                w[j]     = x + y;
                w[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }

    // Max |W|
    let max_coeff = w.iter().map(|v| v.abs()).max().unwrap_or(0);

    // Non-linearity: N_f = 2^(n-1) - 0.5 * max|W|
    let max_non_linearity = (1 << (n_vars - 1)) as f64;
    let actual_non_linearity = max_non_linearity - 0.5 * (max_coeff as f64);
    let non_linearity = if max_non_linearity > 0.0 {
        actual_non_linearity / max_non_linearity
    } else {
        0.0
    };

    // Spectrum entropy over |W|
    let abs_vals: Vec<f64> = w.iter().map(|v| v.abs() as f64).collect();
    let sum_abs: f64 = abs_vals.iter().sum();

    let spectrum_entropy = if sum_abs > 0.0 {
        let mut h_spec = 0.0;
        for &a in &abs_vals {
            if a > 0.0 {
                let p = a / sum_abs;
                h_spec -= p * p.log2();
            }
        }
        h_spec
    } else {
        0.0
    };

    // Plateau detection: how many share max magnitude, and how many unique magnitudes
    use std::collections::HashSet;
    let mut unique_mags: HashSet<i32> = HashSet::new();
    let mut plateau_count = 0usize;

    for &v in &w {
        let mag = v.abs();
        unique_mags.insert(mag);
        if mag == max_coeff {
            plateau_count += 1;
        }
    }

    let unique_magnitudes = unique_mags.len();
    let plateau_ratio = if len > 0 {
        plateau_count as f64 / len as f64
    } else {
        0.0
    };

    // Bent-function distance (heuristic):
    // For a bent function in n_vars (even n), max|W| ≈ 2^(n/2)
    let bent_distance = if n_vars % 2 == 0 {
        let ideal = (1 << (n_vars / 2)) as f64;
        if ideal > 0.0 {
            ((max_coeff as f64 - ideal) / ideal).abs()
        } else {
            0.0
        }
    } else {
        0.0
    };

    WalshMetrics {
        non_linearity,
        spectrum_entropy,
        bent_distance,
        max_coeff,
        plateau_ratio,
        unique_magnitudes,
    }
}















































































pub fn run_tests_no_tracking(stream: &mut BitByteStream) -> bool {
    let n = stream.bit_len;    
    if n < 1_000_000 {
        // too small for research-grade linear complexity
        return false; 
    }    
   
    let mut p = 0.0;   
    p = random_walk_radius_test(stream);                // println!("3D random walk radius test = ");    
    if p < 0.01 { return false; }
	
	p = nist_approximate_entropy_test(stream);          //println!("approximate entropy");        
	if p < 0.01 { return false; }

	p = nist_frequency_test(stream);                    //println!("frequency test");    
	if p < 0.01 { return false; }
	
    p = nist_block_frequency_test(stream);              //println!("block frequency test");    
	if p < 0.01 { return false; }
	
    p = nist_runs_test(stream);                         //println!("runs test");    
	if p < 0.01 { return false; }
	
    p = nist_longest_run_of_ones_test(stream);          //println!("longest run of ones test");    
	if p < 0.01 { return false; }
	
    p = nist_binary_matrix_rank_test(stream);           //println!("binary matrix rank test");    
	if p < 0.01 { return false; }
	
    p = nist_serial_p1_test(stream);                    //println!("serial test 1");    
	if p < 0.01 { return false; }

    p = nist_serial_p2_test(stream);                    //println!("serial test 2");    
    if p < 0.01 { return false; }
		
	p = nist_dft_spectral_test(stream);                 //println!("NIST dft spectral test");    
	if p < 0.01 { return false; }
	
    p = nist_non_overlapping_template_9_test(stream);   //println!("non-overlapping template 9 test");    
	if p < 0.01 { return false; }
	
    p = nist_non_overlapping_template_10_test(stream); //println!("non-overlapping template 10 test");    
	if p < 0.01 { return false; }
	
    p = nist_overlapping_template_test(stream);         //println!("overlapping template test");    
	if p < 0.01 { return false; }
	
    p = nist_universal_maurer_test(stream);             //println!("universal maurer test");    
	if p < 0.01 { return false; }
	
    p = nist_linear_complexity_test(stream);            //println!("linear complexity test");    
	if p < 0.01 { return false; }
	
    p = gap_test(stream);                               //println!("gap test");    
	if p < 0.01 { return false; }
	
    p = nibble_markov_test(stream);                     //println!("nibble markov test");
	if p < 0.01 { return false; }
	
    p = voronoi_cell_volume_test_fast(stream);          //println!("voronoi cell volume test");	
	if p < 0.01 { return false; }
	
    p = ncd_test(stream);                               //println!("NCD history test");    	
	if p < 0.01 { return false; }
	
    p = maurer_universal_byte(stream);             //println!("maurer universal - BYTE test");
	if p < 0.01 { return false; }

    //println!("NIST cumulative sum test");
    p = cusum_forward_test(stream);    
    if p < 0.01 { return false; }
	
	p = cusum_reverse_test(stream);
	if p < 0.01 { return false; }
    
	//this one returns 8-bit entropy values not p-values, use the general P for now...
	//doesn't get logged it's an immediate health check
    //a high the number better... will have to build a custom tracker for this one
    p = entropy_global_test(stream);       
    if p < 7.85 { return false; }
		
	//println!("entropy surface curvature test");
	//returns p and mean
    //let (eP, emP) = entropy_surface_curvature_pvalue(stream);	
    //if eP < 0.01 { return false; }
	
/*
	let panels = [
        // 2-byte words
		(12427, 2),
        (31397, 2),		
        (32749, 2),
		(64513, 2),
        (65521, 2),
        (65537, 2),

        // 3-byte words
        (174763, 3),
        (999431, 3),
        (1048583, 3),
        (1677721, 3),
		(16777199, 3),
        
        // 4-byte words
		(1000000007, 4),
        (2147483647, 4),        
        (3221225473, 4),
		(4294967023, 4),
        (4294967087, 4),
        (4294967291, 4),
    ];
*/

	//let (qP, qmP) = quadratic_character_multi_panel_test(stream, &panels);
	//if qP < 0.01 { return false }

    /*

    let raw1 = nist_random_excursions_test(stream);
    let result1 = aggregate_excursion_panel(raw1);
    if result1.valid_states == 0 {
		return false;
	} else {
		if result1.min_p < 0.01 { return false; }
	}

    let raw2 = nist_random_excursions_variant_test(stream);
    let result2 = aggregate_excursion_panel(raw2);
    if result2.valid_states == 0 {
		return false;
	} else {
		if result2.min_p < 0.01 { return false; }
	}
*/
    true
}



pub fn run_tests_stats(stream: &mut BitByteStream, failed_max: i32, na_max: i32) -> (bool, i32, i32) {    
	let mut failed = 0i32;
	let mut NA = 0i32;
    let n = stream.bit_len;    
    
	if n < 1_000_000 {
        // too small for research-grade linear complexity
        return (false, 0, 0); 
    }    
   
    let mut p = 0.0;   
    p = nist_lempel_ziv_test(stream);
    if p < 0.01 { failed += 1; println!("lempel ziv test test = {}", p); }
	
	p = random_walk_radius_test(stream);
    if p < 0.01 { failed += 1; println!("3D random walk radius test = {}", p); }

    p = gap_test(stream);                               
	if p < 0.01 { failed += 1; println!("gap test = {}", p); }
	
	p = nist_approximate_entropy_test(stream);          
	if p < 0.01 { failed += 1; println!("approximate entropy = {}", p); }
		
	p = nist_frequency_test(stream);                    
	if p < 0.01 { failed += 1; println!("frequency test = {}", p); }
	
    p = nist_block_frequency_test(stream);              
	if p < 0.01 { failed += 1; println!("block frequency test = {}", p); }
	
    p = nist_runs_test(stream);                         
	if p < 0.01 { failed += 1; println!("runs test = {}", p); }
	
    p = nist_longest_run_of_ones_test(stream);          
	if p < 0.01 { failed += 1; println!("longest run of ones test = {}", p); }
		
    p = nist_binary_matrix_rank_test(stream);           
	if p < 0.01 { failed += 1; println!("binary matrix rank test = {}", p); }
	
    p = nist_serial_p1_test(stream);                    
	if p < 0.01 { failed += 1; println!("serial test 1 = {}", p); }

    p = nist_serial_p2_test(stream);                    
    if p < 0.01 { failed += 1; println!("serial test 2 = {}", p); }
			
	p = nist_dft_spectral_test(stream);                 
	if p < 0.01 { failed += 1; println!("NIST dft spectral test = {}", p); }

    p = nist_non_overlapping_template_9_test(stream);   
	if p < 0.01 { failed += 1; println!("non-overlapping template 9 test = {}", p); }

    p = nist_non_overlapping_template_10_test(stream); 
	if p < 0.01 { failed += 1; println!("non-overlapping template 10 test = {}", p); }
	
    p = nist_overlapping_template_test(stream);         
	if p < 0.01 { failed += 1; println!("overlapping template test = {}", p); }

    p = nist_universal_maurer_test(stream);             
	if p < 0.01 { failed += 1; println!("universal maurer test = {}", p); }
 	
    p = nist_linear_complexity_test(stream);            
	if p < 0.01 { failed += 1; println!("linear complexity test = {}", p); }

    p = nibble_markov_test(stream);                     
	if p < 0.01 { failed += 1; println!("nibble markov test = {}", p); }
	
	p = voronoi_cell_volume_calibrated(stream);          
	if p < 0.01 { failed += 1; println!("voronoi cell volume calibrated test = {}", p); }
		
    p = ncd_test(stream);                               
	if p < 0.01 { failed += 1; println!("NCD history test = {}", p); }
    
    p =	maurer_universal_byte(stream);             
	if p < 0.01 { failed += 1; println!("maurer universal calibrated - BYTE test = {}", p); }

    p = cusum_forward_test(stream);                        
    if p < 0.01 { failed += 1; println!("NIST cumulative forward sum test = {}", p); }
	
	p = cusum_reverse_test(stream);                   
	if p < 0.01 { failed += 1; println!("NIST cumulative backward sum test = {}", p); }
    
	//this one returns 8-bit entropy values not p-values, use the general P for now...
	//doesn't get logged it's an immediate health check
    //a high the number better... will have to build a custom tracker for this one
    p = entropy_global_test(stream);                        
    if p < 7.85 { failed += 1; println!("global entropy = {}", p); }
	
	//returns p and mean
    //let (eP, emP) = entropy_surface_curvature_pvalue(stream);     println!("entropy surface curvature test eP = {}  emP = {}", eP, emP);
    //if eP < 0.01 { return false; }
	
	let (qP, qmP, qDevMax, qDevMean) = quadratic_character_multi_panel_metrics(stream);	
	if qP < 0.01 { failed += 1; println!("quadratic panel: lowest p = {}  mean p = {}  max dev = {}  mean dev = {}", qP, qmP, qDevMax, qDevMean); }
	
    let (escP, escPm, escDevMax, escDevMean) = entropy_surface_curvature_multi_panel_metrics(stream);
    if escP < 0.01 { failed += 1; println!("ESC: lowest p = {}  mean p = {}  max dev = {}  mean dev = {}", escP, escPm, escDevMax, escDevMean); }

    // -----------------------
	
    let elig = validate_excursion_eligibility_unified(stream);

	if elig.re_valid {
        let raw = nist_random_excursions_test(elig.j_re as f64, &elig.s_k, &elig.cycle);
		let res = aggregate_excursion_panel(raw);
		if res.min_p < 0.01 { failed += 1; println!("random excursion = {}", res.min_p); }
    } else {
        println!("random excursion NA event");
		NA += 1;
    }

	if elig.rev_valid {
        let raw = nist_random_excursions_variant_test(elig.j_rev as f64, &elig.s_k);
		let res = aggregate_excursion_panel(raw);		
		if res.min_p < 0.01 { failed += 1; println!("random excursion variant = {}", res.min_p); }
    } else {
        println!("random excursion variant NA event");
		NA += 1;
    }

    // -----------------------
   
	if NA > na_max || failed > failed_max { return (false, failed, NA); }
	(true, failed, NA)
}

pub fn write_bits_to_timestamped_file(stream: &BitByteStream) -> std::io::Result<String> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let filename = format!("bits_{}.txt", ts);

    // FIX: unwrap the File
    let mut file = File::create(&filename)?;

    for &bit in &stream.bits {
        let c = if bit == 0 { '0' } else { '1' };
        file.write_all(&[c as u8])?;
    }

    Ok(filename)
}

pub fn write_hex_bytes_to_timestamped_file(stream: &BitByteStream) -> std::io::Result<String> {
    use std::fs::File;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let filename = format!("hexbytes_{}.txt", ts);
    let mut file = File::create(&filename)?;

    // Write each byte as two hex characters
    for &byte in &stream.bytes {
        // format as lowercase hex without 0x prefix
        let hex = format!("{:02x}", byte);
        file.write_all(hex.as_bytes())?;
    }

    Ok(filename)
}

pub fn load_bits_from_file(path: &str) -> BitByteStream {
    // FIX: unwrap the File
    let mut file = File::open(path).expect("Failed to open bit file");

    // FIX: read bytes, not UTF-8 text
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("Failed to read bit file");

    let mut bits = Vec::with_capacity(buf.len());

    for ch in buf {
        match ch {
            b'0' => bits.push(0),
            b'1' => bits.push(1),
            _ => {} // ignore whitespace or stray bytes
        }
    }

    BitByteStream::new_from_bits(bits)
}

/*
fn main() {
    println!("SHA-1 TRUE 48-bit Micro-Universe");

    let max_seeds: u64 = 2000;
    let max_steps: u32 = 150_000_000;

    let mut loops_found = 0u32;
    let mut cycle_lengths: HashSet<u32> = HashSet::new();

    for seed in 0..max_seeds {
        let mut state = seed & ((1u64 << 48) - 1);
        let mut history: HashMap<PackedKey, u32> = HashMap::new();

        let mut loop_found = false;
        let mut loop_len = 0u32;

        for step in 0..max_steps {
            let phase = (step % 12) as u8;
            let key = PackedKey { state, phase };

            if let Some(prev) = history.get(&key) {
                loop_found = true;
                loop_len = step - prev;
                break;
            }

            history.insert(key, step);
            state = sha1_48(state);
        }

        if loop_found {
            loops_found += 1;
            cycle_lengths.insert(loop_len);
            println!("Seed {:05} LOOPED — cycle length {}", seed, loop_len);
        } else {
            println!("Seed {:05} did NOT loop ({max_steps} steps)", seed);
        }
    }
}
*/

/*
fn worker(thread_id: usize) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("long_orbits_thread_{thread_id}.txt"))
        .unwrap();

    let mut highest = 0;

    loop {
        let mut ion = RuntimeIon::new(1, false, 0);
        let mut history: HashMap<PackedKey, u32> = HashMap::new();
        //ion.dump_topology();
		let mut overStepped = true;
        for step in 0..1_000_000 {
            let stream = ion.generate_bits(64);
            let bytes = &stream.bytes;

            let state = ((bytes[0] as u16) << 8) | (bytes[1] as u16);
            let phase = (step % 12) as u8;

            let key = PackedKey { state: state as u64, phase };

            if let Some(prev) = history.get(&key) {
                let loop_len = step - prev;
                if loop_len < 330708 {
				println!("thread {} - loop len {}", thread_id, loop_len);
				
				writeln!(file, "***** Length = [{}]", loop_len).unwrap();
                ion.dump_topology_to(&mut file).unwrap();
				}
                //if loop_len > 1000 {
				//	println!("{}", loop_len);
                //    writeln!(file, "***** Length = [{}]", loop_len).unwrap();
                //    ion.dump_topology_to(&mut file).unwrap();
                //    file.flush().unwrap();
                //}
                overStepped = false;
                break;
            }

            history.insert(key, step);
        }
		if overStepped { println!("loop > 1 million cycles"); }
    }
}
*/

/*
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct DelayState {
    dm0: u64,
    dm1: u64,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct UniverseKey {
    gate: GateType,
    d0: u8,
    d1: u8,
}

struct UniverseInfo {
    id: usize,
    loop_len: usize,
    seed0: u64,
    seed1: u64,
    // map delay state -> orbit index
    states: HashMap<DelayState, usize>,
}

struct UniverseDB {
    next_id: usize,
    universes: HashMap<UniverseKey, Vec<UniverseInfo>>,
}

enum SeedClass {
	Monolith { universe_id: usize },
    DeepHarmonious { universe_id: usize }, 
    NewUniverse { universe_id: usize, loop_len: usize },
    JumpPoint {
        universe_id: usize,
        collided_state: DelayState,
        step: usize,          // local step from seed start
        orbit_index: usize,   // where in the universe’s loop it landed
    },
}

impl UniverseDB {
	fn new() -> Self {
        Self {
            next_id: 0,
            universes: HashMap::new(),
        }
    }
	
    fn classify_seed(&mut self, ion: &mut RuntimeIon) -> SeedClass {
        let key = UniverseKey {
            gate: ion.nodes[0].gate_type,
            d0: ion.nodes[0].delays[0],
            d1: ion.nodes[0].delays[1],
        };

        let entry = self.universes.entry(key).or_insert_with(Vec::new);

        let universe_count = entry.len();       
        if universe_count > 20 {
			let id = self.next_id;			
			return SeedClass::DeepHarmonious { universe_id: id};
        }

        let mut local_history: HashMap<DelayState, usize> = HashMap::new();
        let mut step: usize = 0;

		loop {
			// 1. tick FIRST
			let _bit = ion.generate_bit();
		
			// 2. read delay masks AFTER tick
			let node = &ion.nodes[0];
			let state = DelayState {
				dm0: node.delay_masks[0],
				dm1: node.delay_masks[1],
			};

			// 3. check collision with existing universes
			for uni in entry.iter() {
				if let Some(&orbit_index) = uni.states.get(&state) {
					return SeedClass::JumpPoint {
						universe_id: uni.id,
						collided_state: state,
						step,
						orbit_index,
					};
				}
			}
            
			// 4. check local loop closure			
			if let Some(first_step) = local_history.get(&state) {
				let loop_len = step - *first_step;

				let id = self.next_id;
				self.next_id += 1;
				
				let mut new_states = HashMap::new();
				// local_history: state -> first_seen_step
				// we want orbit index = (first_seen_step - *first_step) mod loop_len
				for (s, s_step) in local_history.iter() {
					let orbit_index = (*s_step - *first_step) % loop_len;
					new_states.insert(*s, orbit_index);
				}

				entry.push(UniverseInfo {
					id,
					loop_len,
					seed0: ion.nodes[0].initial_delay_masks[0],
					seed1: ion.nodes[0].initial_delay_masks[1],
					states: new_states,
				});

				return SeedClass::NewUniverse {
					universe_id: id,
					loop_len,
				};
			}


			// 5. record state
			local_history.insert(state, step);

			// 6. increment step detect the monolith 
			step += 1;
			if step > 25_000_000 {
                let id = self.next_id;
				self.next_id += 1;
				let mut new_states = HashMap::new();

				// local_history: state -> first_seen_step
				// we want orbit index = (first_seen_step - *first_step) mod loop_len
				for (s, s_step) in local_history.iter() {
					let orbit_index = *s_step;
					new_states.insert(*s, orbit_index);
				}
				entry.push(UniverseInfo {
					id,
					loop_len: 3_735_928_559,  // 0xDEADBEEF easy identifer/number since we don't track > 250 million
					seed0: ion.nodes[0].initial_delay_masks[0],
					seed1: ion.nodes[0].initial_delay_masks[1],
					states: new_states,
				});
				return SeedClass::Monolith { universe_id: id};
			}
		}
    }
}
*/

//use std::collections::HashMap;

const BUCKETS: usize = 23;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct DelayState {
    pub dm0: u64,
    pub dm1: u64,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct UniverseKey {
    pub gate: GateType,
    pub d0: u8,
    pub d1: u8,
}

pub struct UniverseInfo {
    pub id: usize,
    pub loop_len: usize,
    pub seed0: u64,
    pub seed1: u64,
}

pub struct UniverseBucket {
    pub universes: Vec<UniverseInfo>,
    // 23 buckets: DelayState -> (universe_id, orbit_index)
    pub states: Vec<HashMap<DelayState, (usize, usize)>>,
}

pub struct UniverseDB {
    pub next_id: usize,
    pub universes: HashMap<UniverseKey, UniverseBucket>,
}

pub enum SeedClass {
    Monolith { universe_id: usize },
    DeepHarmonious { universe_id: usize },
    NewUniverse { universe_id: usize, loop_len: usize },
    JumpPoint {
        universe_id: usize,
        collided_state: DelayState,
        step: usize,
        orbit_index: usize,
    },
}

impl UniverseDB {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            universes: HashMap::new(),
        }
    }
}

fn bucket_index(s: &DelayState) -> usize {
    let h = s.dm0.wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ s.dm1.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (h % BUCKETS as u64) as usize
}

impl UniverseDB {
    pub fn classify_seed(&mut self, ion: &mut RuntimeIon) -> SeedClass {
        let key = UniverseKey {
            gate: ion.nodes[0].gate_type,
            d0: ion.nodes[0].delays[0],
            d1: ion.nodes[0].delays[1],
        };

        // Create bucket if missing
        let bucket = self.universes.entry(key).or_insert_with(|| {
            let mut states = Vec::with_capacity(BUCKETS);
            for _ in 0..BUCKETS {
                states.push(HashMap::new());
            }
            UniverseBucket {
                universes: Vec::new(),
                states,
            }
        });

        //let universe_count = bucket.universes.len();
        //if universe_count > 20 {
        //    let id = self.next_id;
        //    return SeedClass::DeepHarmonious { universe_id: id };
        //}

        let mut local_history: HashMap<DelayState, usize> = HashMap::new();
        let mut step: usize = 0;

        loop {
            // 1. tick
            let _bit = ion.generate_bit();

            // 2. read delay masks
            let node = &ion.nodes[0];
            let state = DelayState {
                dm0: node.delay_masks[0],
                dm1: node.delay_masks[1],
            };

            // 3. O(1) bucketed collision check
            let b = bucket_index(&state);
            if let Some(&(universe_id, orbit_index)) = bucket.states[b].get(&state) {
                return SeedClass::JumpPoint {
                    universe_id,
                    collided_state: state,
                    step,
                    orbit_index,
                };
            }

            // 4. local loop closure
            if let Some(first_step) = local_history.get(&state) {
                let loop_len = step - *first_step;

                let id = self.next_id;
                self.next_id += 1;

                // register universe metadata
                bucket.universes.push(UniverseInfo {
                    id,
                    loop_len,
                    seed0: ion.nodes[0].initial_delay_masks[0],
                    seed1: ion.nodes[0].initial_delay_masks[1],
                });

                // populate shared bucketed state map
                for (s, s_step) in local_history.iter() {
                    let orbit_index = (*s_step - *first_step) % loop_len;
                    let sb = bucket_index(s);
                    bucket.states[sb].insert(*s, (id, orbit_index));
                }

                return SeedClass::NewUniverse {
                    universe_id: id,
                    loop_len,
                };
            }

            // 5. record state
            local_history.insert(state, step);

            // 6. increment step
            step += 1;
        }
    }
}

/*
fn probe_monolith(
    ion: &mut RuntimeIon,
    prefix_len: usize,
    max_steps: usize,
) -> Option<usize> {

    // --- Capture prefix frames ---
    let mut prefix: Vec<DelayState> = Vec::with_capacity(prefix_len);

    for _ in 0..prefix_len {
        let _bit = ion.generate_bit();
        let node = &ion.nodes[0];
        prefix.push(DelayState {
            dm0: node.delay_masks[0],
            dm1: node.delay_masks[1],
        });
    }

    // --- Run forward until match or timeout ---
    let mut steps = prefix_len;

    loop {
        let _bit = ion.generate_bit();
        let node = &ion.nodes[0];
        let state = DelayState {
            dm0: node.delay_masks[0],
            dm1: node.delay_masks[1],
        };

        steps += 1;

        // match against prefix frames
        if let Some(idx) = prefix.iter().position(|p| *p == state) {
            return Some(steps - idx);
        }

        // monster: no repeat found
        if steps >= max_steps {
            return None;
        }
    }
}

fn monolithWorker(thread_id: usize, d1: u8, d2: u8, gate: GateType) {
    // 1. Build the engine
    let mut ion = RuntimeIon::new(
                1,
                1,
                d1,
                d2,
                0x0000000000000001,
                0x0000000000000001,
                gate
            );

    // Probe the monolith
    match probe_monolith(&mut ion, 1000, 50_000_000_000) {
        Some(len) => println!("monolith loop_len = {}", len),
        None => println!("monster: loop_len > 50 billion"),
    }
}
*/

fn worker(thread_id: usize, d1: u8, d2: u8, gate: GateType) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("universe_map_thread_{thread_id}.txt"))
        .unwrap();

    let mut universe_db = UniverseDB::new();

    // Sweep seeds
    for i in 0..64 {
        let seed0: u64 = 1u64 << i;

        for j in 0..64 {
            let seed1: u64 = 1u64 << j;

            let mut ion = RuntimeIon::new(
                1,
                1,
                d1,
                d2,
                seed0,
                seed1,
                gate
            );

            // Warmup if needed
            for _ in 0..64 {
                ion.generate_bit();
            }

			match universe_db.classify_seed(&mut ion) {
				SeedClass::Monolith { universe_id } => {
					writeln!(
						file,
						"MONOLITH\tid={}\tgate={:?}\tdelays=({}, {})\tseed0=0x{:016X}\tseed1=0x{:016X}\tloop_len=capped >250_000_0000",
						universe_id,
						gate,
						ion.nodes[0].delays[0],
						ion.nodes[0].delays[1],
						seed0,
						seed1						
					).unwrap();
					
					println!(
						"thread {}: NEW monolith id={} seed0=0x{:X} seed1=0x{:X}",
						thread_id,
						universe_id,						
						seed0,
						seed1
					);
					file.flush().unwrap();
					return;
			    }
				
				SeedClass::DeepHarmonious { universe_id} => {
					writeln!(
						file,
						"DEEP HARMONIOUS - ABORTED",
					).unwrap();
					
					println!(
						"thread {}: DEEP HARMONIOUS - ABORTING",
						thread_id,						
					);
					file.flush().unwrap();
					return;
			    }	
				
				SeedClass::NewUniverse { universe_id, loop_len } => {
					writeln!(
						file,
						"NEW_UNIVERSE\tid={}\tgate={:?}\tdelays=({}, {})\tseed0=0x{:016X}\tseed1=0x{:016X}\tloop_len={}",
						universe_id,
						gate,
						ion.nodes[0].delays[0],
						ion.nodes[0].delays[1],
						seed0,
						seed1,
						loop_len
					).unwrap();
                    /*
					println!(
						"thread {}: NEW universe id={} loop_len={} seed0=0x{:X} seed1=0x{:X}",
						thread_id,
						universe_id,
						loop_len,
						seed0,
						seed1
					);*/
				}
                
				SeedClass::JumpPoint { universe_id, collided_state, step, orbit_index } => {										
					writeln!(
						file,
						"JUMP_POINT\tinto_universe={}\tgate={:?}\tdelays=({}, {})\tseed0=0x{:016X}\tseed1=0x{:016X}\tat_step={}\torbit_index={}\tcollided_state=(0x{:016X},0x{:016X})",
						universe_id,
						gate,
						ion.nodes[0].delays[0],
						ion.nodes[0].delays[1],
						seed0,
						seed1,
						step,
						orbit_index,
						collided_state.dm0,
						collided_state.dm1
						).unwrap();
                    /*
					println!(
						"thread {}: JUMP into universe {} at local_step={} orbit_index={} seed0=0x{:X} seed1=0x{:X} collided_state=(0x{:X},0x{:X})",
						thread_id,
						universe_id,
						step,
						orbit_index,
						seed0,
						seed1,
						collided_state.dm0,
						collided_state.dm1
					);*/
				}
			}
			
			file.flush().unwrap();
		}
	}
}

/*
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PackedKey {
    state: u64, // we only use lower 48 bits
    phase: u16,
}

fn worker(thread_id: usize, gate: GateType) {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::collections::HashMap;

    let mut ion = RuntimeIon::new(1, 1, 2, 16, 0x0000000000000002, 0x0000000000009FC8, GateType::AND);
	let stream = ion.generate_bits(389_283_968);
	write_hex_bytes_to_timestamped_file(&stream);
    return;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("good_attractors_thread_{thread_id}.txt"))
        .unwrap();
	
	// foo
	// 17/19 = 60 bit reads must be in blocks of 60 bits or sync is lost
	// 2/16 = 64 any even order of power of 2.  cannot be odd reads.
	
    for foo in 60..=60  {
        for i in 0..64 {
            let seed0: u64 = 1u64 << i;
            for j in 0..64 {
                let seed1: u64 = 1u64 << j;
				//(node_count: usize, state: u64, delay0: u8, delay1: u8, delay_mask0: u64, delay_mask1: u64, gate: GateType) -> Self {
                let mut ion = RuntimeIon::new(1, 1, 17, 19, seed0, seed1, gate);
		        let stream = ion.generate_bits(foo * 8);	
		        let node = &ion.nodes[0];
		        let postWUSeed0 = node.delay_masks[0];
                let postWUSeed1 = node.delay_masks[1];
        
                let mut history: HashMap<PackedKey, usize> = HashMap::new();
                let mut out = [0u8; 1];

                let mut found = false;
                let mut loop_len: usize = 0;

                for step in 0..99_000_000 {
                    let mut stream = ion.generate_bits(foo * 8);
                    let mut bytes = &stream.bytes;
            
                    let state = ((bytes[0] as u16) << 8) | (bytes[1] as u16);
                    let phase = (step % 12) as u16;

                    let key = PackedKey {
                        state: state as u64,
                        phase,
                    };

                    if let Some(prev) = history.get(&key) {
                        loop_len = step - *prev;
                        found = true;
                        break;
                    }

                    history.insert(key, step);
                }

                if !found {
					loop_len = 0xDEADBEEF;
                    println!("thread {}: loop > 1M", thread_id);
                    continue;
                }
        		
                let node = &ion.nodes[0];
                let seed0 = node.initial_delay_masks[0];
                let seed1 = node.initial_delay_masks[1];
         		let currentSeed0 = node.delay_masks[0];
                let currentSeed1 = node.delay_masks[1];
        	
                let d0 = node.delays[0];
                let d1 = node.delays[1];
                let gate = node.gate_type;
                let state = node.state;

                writeln!(
                    file,
                    "loop_len={}\tseed0=0x{:016X}\tseed1=0x{:016X}\tdelays=({}, {})\tgate={:?}\tstate={}\n\t\tseed0=0x{:016X}\tseed1=0x{:016X}",
                    loop_len, postWUSeed0, postWUSeed1, d0, d1, gate, state, currentSeed0, currentSeed1,
                ).unwrap();

                file.flush().unwrap();

                println!(
                    "thread {}: GOOD attractor {} seeds (0x{:X}, 0x{:X})",
                    thread_id, loop_len, seed0, seed1
                );
            }
        }
	}
}
*/

/*
fn main() {
    //  monolithWorker(9999,2,29, GateType::OR/AND);  monolith loop_len = 536870909
    
	// this whole band is likely a giant monolith, and sub-divisions of it... 
	//  monolithWorker(9999,1,34, GateType::OR);      monolith loop_len = 5637144493

    // monolithWorker(9999,1,37, GateType::OR);  didn't complete after running for 8 hours.  	
    	
    let mut handles = Vec::new();
    let mut threadID = 1000;

    for i in 1..=62 {
        for j in (i + 1)..=63 {

            // Spawn 4 threads per (i,j)
            threadID += 1;
            handles.push(thread::spawn(move || worker(threadID, i, j, GateType::NOR)));

            threadID += 1;
            handles.push(thread::spawn(move || worker(threadID, i, j, GateType::AND)));

            threadID += 1;
            handles.push(thread::spawn(move || worker(threadID, i, j, GateType::OR)));

            threadID += 1;
            handles.push(thread::spawn(move || worker(threadID, i, j, GateType::NAND)));

            // Flush every 24 threads
            if handles.len() >= 24 {
                for h in handles.drain(..) {
                    match h.join() {
                        Ok(_) => {}
                        Err(e) => {
                           println!("Thread {} panicked: {:?}", threadID, e);
                        }
                    }
                }
            }
        }
    }

    // Flush any remaining threads at the end
    for h in handles {
        match h.join() {
            Ok(_) => {}
            Err(e) => {
                 println!("Thread {} panicked: {:?}", threadID, e);
            }
        }
    }
}
*/

//use std::fs::File;
use std::io::{BufRead, BufReader};
//use std::thread;

const MAX_THREADS: usize = 2;

fn main() {
    // Load splinter tasks
    let file = File::open("splintered.txt")
        .expect("splintered.txt not found");
    let reader = BufReader::new(file);

    // Parsed tasks stored here
    let mut tasks = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        if line.contains('#') { continue; }
		if !line.contains("thread=") { continue; }

        // Parse thread ID
        let thread_id: usize = line
            .split_whitespace()
            .find(|s| s.starts_with("thread="))
            .and_then(|s| s[7..].parse().ok())
            .expect("Failed to parse thread ID");

        // Parse gate
        let gate_str = line
            .split_whitespace()
            .find(|s| s.starts_with("gate="))
            .map(|s| &s[5..])
            .expect("Failed to parse gate");

        let gate = match gate_str {
            "AND" => GateType::AND,
            "OR"  => GateType::OR,
            "NOR" => GateType::NOR,
            "NAND"=> GateType::NAND,
            _ => panic!("Unknown gate type {}", gate_str),
        };

        // Parse delays=(x, y)
        let delays_start = line.find("delays=(")
            .expect("Failed to find delays=(") + "delays=(".len();
        let delays_end = line[delays_start..]
            .find(')')
            .expect("Failed to find closing ) for delays")
            + delays_start;

        let delays_str = &line[delays_start..delays_end];
        let mut nums = delays_str.split(',');

        let d0: u8 = nums.next()
            .expect("Missing first delay")
            .trim()
            .parse()
            .expect("Failed to parse first delay");
        let d1: u8 = nums.next()
            .expect("Missing second delay")
            .trim()
            .parse()
            .expect("Failed to parse second delay");

        tasks.push((thread_id, d0, d1, gate));
    }

    println!("Loaded {} splinter tasks.", tasks.len());

    // Thread pool (manual)
    let mut active_threads: Vec<thread::JoinHandle<()>> = Vec::new();
    let mut task_index = 0;

    while task_index < tasks.len() || !active_threads.is_empty() {
        // Launch new threads if we have capacity
        while active_threads.len() < MAX_THREADS && task_index < tasks.len() {
            let (thread_id, d0, d1, gate) = tasks[task_index];
            task_index += 1;

            println!(
                ">>> Launching splinter worker: thread={} gate={:?} delays=({}, {})",
                thread_id, gate, d0, d1
            );

            let handle = thread::spawn(move || {
                worker(thread_id, d0, d1, gate);
                println!(
                    "<<< Completed splinter worker: thread={} gate={:?} delays=({}, {})",
                    thread_id, gate, d0, d1
                );
            });

            active_threads.push(handle);
        }

        // Check for finished threads
        let mut i = 0;
        while i < active_threads.len() {
            if active_threads[i].is_finished() {
                let handle = active_threads.remove(i);
                handle.join().unwrap();
            } else {
                i += 1;
            }
        }

        // Small sleep to avoid busy-waiting
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    println!("All splintered entries processed with {} parallel workers.", MAX_THREADS);
}

/*
fn main() {    
    //let mut stream = load_bits_from_file("bits_1784562079.txt");
    //run_tests_stats(&mut stream);
    //return;
    let mut old_highest = 0;
    loop {
        // fresh engine every iteration
        let mut ion = RuntimeIon::new(1, false, 0);

        let mut history: HashMap<PackedKey, u32> = HashMap::new();
		let mut loop_found = false;
        let mut loop_len = 0u32;
		
        for step in 0..44_000_000 {
            let stream = ion.generate_bits(100_000);
            let bytes = &stream.bytes;

            let state = ((bytes[0] as u16) << 8) | (bytes[1] as u16);
            let phase = (step % 12) as u8;

            let key = PackedKey { state: state as u64, phase };

            if let Some(prev) = history.get(&key) {
                loop_found = true;
                loop_len = step - prev;
				if loop_len > 1500 { println!("***** Length = [{}]", loop_len); ion.dump_topology(); }				
                break;
            }

            history.insert(key, step);          
        }
        continue;

        // warm-up + generate full stream
        ion.generate_bits(100_000);
        let mut stream = ion.generate_bits(1_000_000);

        // run LC test
        let (passed, _, _)  = run_tests_stats(&mut stream, 0, 0);

        // success condition
        if passed {
            //run_tests_stats(&mut stream);
            println!("=== FOUND A LINEAR-COMPLEXITY PASS ===");            
            match write_bits_to_timestamped_file(&stream) {
                Ok(name) => println!("Wrote {}", name),
                Err(e) => eprintln!("Error: {}", e),
            } 
            stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_stats(&mut stream));
            let (mut result, mut failed, mut na) = run_tests_stats(&mut stream, 4, 2);
            if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
			stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
            if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
			stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
            if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
			stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
            if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
			stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
            if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
			stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
			if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
            stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
			if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
            stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
			if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
            stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
			if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }
			
            stream = ion.generate_bits(1_000_000); //println!("{}", run_tests_no_tracking(&mut stream));
            (result, failed, na) = run_tests_stats(&mut stream, 4, 2);
			if result == false { println!("aborting failed..."); continue; } else { println!("Passed F=[{}] NA=[{}]", failed, na); }

            ion.dump_topology();

            match write_bits_to_timestamped_file(&stream) {
                Ok(name) => println!("Wrote {}", name),
                Err(e) => eprintln!("Error: {}", e),
            }

            break; // stop the endless loop
        } else { println!("Failed!"); }
    }
}
*/
