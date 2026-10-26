//! fx.cu on the GPU (cargo build --features gpu): the pixel work of a batch of pictures, through CUDA.
//! The kernels are compiled at start by NVRTC (libnvrtc.so.13 on LD_LIBRARY_PATH, e.g. PyTorch's copy
//! under .venv/.../nvidia/cu13/lib) and libcuda comes with the driver (in WSL: /usr/lib/wsl/lib).

use crate::look::{cuda_defines, DEBRIS_W, MAX_DEBRIS, NP, P_DEFOCUS, P_MASK_BLUR};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

pub struct Gpu {
    stream: Arc<CudaStream>,
    warp_chalk: CudaFunction,
    max_filter: CudaFunction,
    blur: CudaFunction,
    colour: CudaFunction,
    motion_noise: CudaFunction,
    lowres_bytes: CudaFunction,
    pub name: String,
}

fn err<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let ctx = CudaContext::new(0).map_err(err)?;
        let name = ctx.name().map_err(err)?;
        let src = cuda_defines() + include_str!("fx.cu");
        let ptx = compile_ptx_with_opts(src, CompileOptions { name: Some("fx.cu".into()), ..Default::default() }).map_err(err)?;
        let m = ctx.load_module(ptx).map_err(err)?;
        let f = |n: &str| m.load_function(n).map_err(err);
        Ok(Gpu {
            stream: ctx.default_stream(),
            warp_chalk: f("warp_chalk")?,
            max_filter: f("max_filter")?,
            blur: f("blur")?,
            colour: f("colour")?,
            motion_noise: f("motion_noise")?,
            lowres_bytes: f("lowres_bytes")?,
            name,
        })
    }

    /// b pictures: masks (b*s*s), params (b*NP), debris (b*MAX_DEBRIS*DEBRIS_W) -> RGB bytes (b*s*s*3).
    pub fn run(&self, masks: &[u8], params: &[f32], debris: &[f32], s: usize, b: usize) -> Result<Vec<u8>, String> {
        assert_eq!(params.len(), b * NP);
        assert_eq!(debris.len(), b * MAX_DEBRIS * DEBRIS_W);
        let st = &self.stream;
        let n = s * s;
        let (si, bi) = (s as i32, b as i32);
        let m = st.clone_htod(masks).map_err(err)?;
        let p = st.clone_htod(params).map_err(err)?;
        let d = st.clone_htod(debris).map_err(err)?;
        let mut a = st.alloc_zeros::<f32>(b * n).map_err(err)?;
        let mut a2 = st.alloc_zeros::<f32>(b * n).map_err(err)?;
        let mut c = st.alloc_zeros::<f32>(3 * b * n).map_err(err)?;
        let mut c2 = st.alloc_zeros::<f32>(3 * b * n).map_err(err)?;
        let mut out = st.alloc_zeros::<u8>(3 * b * n).map_err(err)?;
        let one = LaunchConfig::for_num_elems((b * n) as u32);
        let three = LaunchConfig::for_num_elems((3 * b * n) as u32);
        let (h, v, one_plane, three_planes) = (0i32, 1i32, 1i32, 3i32);
        let (mask_blur, defocus) = (P_MASK_BLUR as i32, P_DEFOCUS as i32);
        unsafe {
            let mut l = st.launch_builder(&self.warp_chalk);
            l.arg(&m).arg(&p).arg(&mut a).arg(&si).arg(&bi);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.max_filter);
            l.arg(&a).arg(&mut a2).arg(&p).arg(&si).arg(&bi).arg(&h);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.max_filter);
            l.arg(&a2).arg(&mut a).arg(&p).arg(&si).arg(&bi).arg(&v);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.blur);
            l.arg(&a).arg(&mut a2).arg(&p).arg(&mask_blur).arg(&one_plane).arg(&si).arg(&bi).arg(&h);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.blur);
            l.arg(&a2).arg(&mut a).arg(&p).arg(&mask_blur).arg(&one_plane).arg(&si).arg(&bi).arg(&v);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.colour);
            l.arg(&a).arg(&p).arg(&d).arg(&mut c).arg(&si).arg(&bi);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.motion_noise);
            l.arg(&c).arg(&mut c2).arg(&p).arg(&si).arg(&bi);
            l.launch(one).map_err(err)?;
            let mut l = st.launch_builder(&self.blur);
            l.arg(&c2).arg(&mut c).arg(&p).arg(&defocus).arg(&three_planes).arg(&si).arg(&bi).arg(&h);
            l.launch(three).map_err(err)?;
            let mut l = st.launch_builder(&self.blur);
            l.arg(&c).arg(&mut c2).arg(&p).arg(&defocus).arg(&three_planes).arg(&si).arg(&bi).arg(&v);
            l.launch(three).map_err(err)?;
            let mut l = st.launch_builder(&self.lowres_bytes);
            l.arg(&c2).arg(&mut out).arg(&p).arg(&si).arg(&bi);
            l.launch(one).map_err(err)?;
        }
        st.clone_dtoh(&out).map_err(err)
    }
}
