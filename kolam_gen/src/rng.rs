//! xoshiro256** seeded from (seed, index, attempt): every sample is reproducible on its own.
//! Same draws as the Python generator uses (numpy Generator), not the same stream.

pub struct Rng {
    s: [u64; 4],
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64, index: u64, attempt: u64) -> Rng {
        let mut x = seed;
        let a = splitmix(&mut x);
        let mut x = a ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03);
        let b = splitmix(&mut x);
        let mut x = b ^ attempt.wrapping_mul(0x8CB9_2BA7_2F3D_8DD7);
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            *v = splitmix(&mut x);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform in lo..hi (numpy integers(lo, hi)).
    pub fn integers(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo) as usize) as i32
    }

    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// An index drawn with weights p (need not sum to 1).
    pub fn choice_p(&mut self, p: &[f64]) -> usize {
        let total: f64 = p.iter().sum();
        let mut x = self.random() * total;
        for (i, &w) in p.iter().enumerate() {
            if x < w {
                return i;
            }
            x -= w;
        }
        p.iter().rposition(|&w| w > 0.0).unwrap_or(0)
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
        v
    }

    /// dirichlet(ones(k)): normalised exponentials.
    pub fn dirichlet_ones(&mut self, k: usize) -> Vec<f64> {
        let e: Vec<f64> = (0..k).map(|_| -(1.0 - self.random()).ln()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect()
    }
}
