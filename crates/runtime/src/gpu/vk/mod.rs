//! Vulkan device context shared by the renderer and the presenter.

pub mod defer;
pub mod framewatch;
pub mod pbin;
pub mod present;
pub mod sharpen;
#[cfg_attr(target_os = "linux", path = "../../platform/linux/window.rs")]
pub mod window;

// Preserve the existing presence switch, but an explicit zero disables validation.
fn validation_requested(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v != "0")
}

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, c_char};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Ctx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: Mutex<vk::Queue>,
    pub queue_family: u32,
    /// A second queue of the graphics family, for the hidden present of runs without a window
    /// (`present::Hidden`) alone: its submissions never wait behind the main queue's work and
    /// its presents never hold the main queue's lock (None: the family has one queue).
    pub present_queue: Option<Mutex<vk::Queue>>,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub surface_i: ash::khr::surface::Instance,
    #[cfg(windows)]
    pub win32_i: ash::khr::win32_surface::Instance,
    pub swapchain_d: ash::khr::swapchain::Device,
    pub push_desc: ash::khr::push_descriptor::Device,
    /// VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors.
    pub max_push: u32,
    /// VK_EXT_external_memory_host (zero-copy access to guest memory).
    pub ext_host: Option<ash::ext::external_memory_host::Device>,
    /// The device accepts dma-buf imports (RADV does, for memfd sections).
    pub dmabuf: bool,
    /// VK_NV_device_diagnostic_checkpoints (DS_GPU_CHECKPOINTS=1): markers
    /// read back after a device loss.
    pub checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    /// VK_KHR_pipeline_executable_properties (DS_PIPELINE_STATS=1, `gpu::pstats`).
    pub exec_props: Option<ash::khr::pipeline_executable_properties::Device>,
    pub device_name: String,
    pub validation: bool,
    /// VK_EXT_image_view_min_lod on: sampled views carry the T#'s MIN_LOD (`images`).
    pub view_min_lod: bool,
    /// Vulkan 1.2's hostQueryReset enabled: batch slots' timestamp queries
    /// are reset by the CPU when the slot is released (`exec::Slot::bq`).
    pub host_query_reset: bool,
    /// VK_EXT_debug_utils command labels (DS_GPU_LABELS=1): named ranges of
    /// the recorded work for GPU Trace (Nsight Graphics).
    pub labels: Option<ash::ext::debug_utils::Device>,
    /// VK_EXT_debug_utils object names under the validation layer
    /// (DS_VK_VALIDATION=1): its messages name the objects (`name_object`).
    pub names: Option<ash::ext::debug_utils::Device>,
    /// VK_KHR_fragment_shader_barycentric enabled.
    pub barycentric: bool,
    /// Without it: pixel stages that read per vertex get each vertex's
    /// values from a generated geometry shader (`rdna::translate::Spec::bary_gs`).
    pub bary_gs: bool,
    /// VK_NV_shader_subgroup_partitioned enabled (`subgroup_partitioned`).
    pub partitioned: bool,
    /// VK_EXT_memory_budget enabled (`heap_report`).
    pub memory_budget: bool,
    /// The device's default subgroup size (32 on NVIDIA; AMD 64, Intel 8-32).
    pub subgroup_size: u32,
    /// Stages given 32-lane subgroups (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`):
    /// the translation's cross-lane code is written for subgroups of 32 (`subgroup32`).
    pub subgroup32: vk::ShaderStageFlags,
    /// VK_EXT_calibrated_timestamps with the device and performance-counter
    /// time domains (`gpu::timeline`: GPU timestamps on the CPU clock).
    pub calibrated: Option<ash::ext::calibrated_timestamps::Device>,
    /// VK_EXT_device_fault: the device's own account of a loss (`fault_report`).
    pub fault: Option<ash::ext::device_fault::Device>,
    /// VK_EXT_full_screen_exclusive (`present`): exclusive fullscreen acquired by the
    /// presenter, and windows the driver must not make exclusive on its own.
    pub fse: Option<ash::ext::full_screen_exclusive::Device>,
    /// VK_EXT_swapchain_colorspace: HDR10 (ST 2084) swapchains where the display offers them.
    pub colorspaces: bool,
    /// VK_EXT_hdr_metadata: the HDR10 mastering metadata of an HDR swapchain.
    pub hdr_metadata: Option<ash::ext::hdr_metadata::Device>,
    /// bufferDeviceAddress enabled: guest imports and video-memory homes get
    /// GPU addresses that fault reports name (`note_gpu_range`).
    pub device_addresses: bool,
    /// robustBufferAccess2 and nullDescriptor enabled: loads past a storage
    /// buffer binding's range read 0, and a null buffer binds as all zeros
    /// (`tune hw_bounds`).
    pub robust2_null: bool,
    /// drawIndirectCount enabled: batched draws grow after recording
    /// (`exec::draw` draw runs).
    pub draw_indirect_count: bool,
    /// sparseBinding and sparseResidencyBuffer enabled (`DS_RESIDENT=1`:
    /// the residency mirror, `gpu::resident`).
    pub sparse_buffers: bool,
    /// shaderDrawParameters and drawIndirectFirstInstance enabled: vertex
    /// shaders of indirect draws take the start instance from
    /// gl_BaseInstance (`tune cpu_args_base`).
    pub draw_parameters: bool,
    /// The copy queue (a transfer-only queue family; `DS_COPY_QUEUE=0`:
    /// none): batch-start copies run on it beside the main queue (`exec`).
    pub copy: Option<CopyQueue>,
    /// Ordered async compute (RENDERER.md; `DS_ASYNC_COMPUTE` 2 or more at
    /// boot, `async_compute_boot`): a queue of a compute-only family that the
    /// compute queues' batches run on beside the main queue.
    pub compute: Option<ComputeQueue>,
    /// With `compute`: a timeline every submission to the main queue
    /// signals with the next value of `g_count` (`progress_signal`); a batch
    /// on the compute queue waits for the value current when it is
    /// submitted, after all earlier work of the main queue.
    pub g_progress: vk::Semaphore,
    g_count: AtomicU64,
    /// The distinct queue families buffers are used by (the main queue's,
    /// the copy queue's, the compute queue's).
    pub families: Vec<u32>,
    /// A timeline semaphore each batch on the main queue signals with its
    /// sequence (`exec`: batches submitted together, completions waited for
    /// by value).
    pub batches: vk::Semaphore,
    /// A timeline semaphore the host signals with a batch's sequence once
    /// its deferred effects (CP writes the CPU performs) ran: batches whose
    /// GPU-performed CP writes follow those wait for it
    /// (`exec::Exec::order_after_cpu`).
    pub cpu_done: vk::Semaphore,
    /// VK_KHR_pipeline_binary where the device has it (`pipeline_binaries`):
    /// the pipeline store keeps the driver's binaries (`gpu::pcache`).
    pub pbin: Option<pbin::Fns>,
}

/// The pipeline store keeps the driver's pipeline binaries (VK_KHR_pipeline_binary,
/// `gpu::pcache`) where the device has them (since 2026-11-08; `DS_PIPELINE_BINARIES=0`: cache
/// data only); read when the device is made. Binaries hold the compiled code themselves; the
/// store's cache data does not on this driver: on 2026-11-21 night two boots on a store of cache
/// data compiled every creation (1.4 s each, 1,834-1,903 of them), with the driver's own cache as
/// it was and with it in an empty folder, where at 18:00 the same store had served in 1.3 ms
/// (after other runs had filled the driver's cache, which is shared with every program); on
/// 2026-11-08 a reset of that cache did the same. (A store check's four small pipelines did serve
/// from cache data alone: not the real ones.) Some drivers refuse binaries across processes (the
/// other PC's refused 2,136 of 2,670 that night): the driver's own cache stays on beside them
/// since then (`DS_PB_NO_INTERNAL=1` turns it off, as before), so a refused binary's compile is
/// served from it when it holds the pipeline, as cache data had been there.
pub fn pipeline_binaries() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PIPELINE_BINARIES").ok().as_deref() != Some("0"))
}



/// A queue of a compute-only family (`Ctx::compute`) and the timeline its
/// batches signal with their sequence.
pub struct ComputeQueue {
    pub family: u32,
    pub queue: Mutex<vk::Queue>,
    pub batches: vk::Semaphore,
}

/// Ordered async compute is set up at boot (`DS_ASYNC_COMPUTE` 2 or more
/// when the device is made): the compute family's queue, the second
/// written-bit map, images shared between the families.
pub fn async_compute_boot() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| super::tune::async_compute() >= 2)
}

/// A transfer-only queue for batch-start copies, and its timeline.
pub struct CopyQueue {
    pub family: u32,
    pub queue: Mutex<vk::Queue>,
    /// Timeline semaphore: reaches a batch's sequence once the copies it
    /// made on this queue are done (signalled in sequence order).
    pub copied: vk::Semaphore,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

/// The chosen device is one the translation's subgroup modes were checked on
/// (`shaders::plain_modes`); set when the device is chosen.
static SUBGROUP_MODES_CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the chosen device is a design the translation's subgroup modes were checked on: NVIDIA's
/// before Blackwell (PCI device ids below 0x2B80: the RTX 40 series and older; the RTX 50 series
/// starts at 0x2B80). Their tests ran on an RTX 4090 and an RTX 4070 SUPER.
pub fn subgroup_modes_checked() -> bool {
    SUBGROUP_MODES_CHECKED.load(std::sync::atomic::Ordering::Relaxed)
}

fn modes_checked_on(vendor: u32, device: u32) -> bool {
    // On Linux the subgroup modes were also checked on a Blackwell card (an RTX PRO 6000,
    // 10de:2bb1, driver 615.78): with them and rdna's NaN handling the tutorial and the tunnel
    // render right, while the portable translation drew the 3D scene as scattered coloured
    // specks. So NVIDIA keeps the modes on Linux whatever the design (DS_PLAIN_MODES overrides).
    vendor == 0x10de && (device < 0x2b80 || cfg!(target_os = "linux"))
}

// Subgroup control flow and floating-point preservation are independent policies. Linux
// validates NVIDIA subgroup modes even on Blackwell; that must not disable its NaN guard.
fn nan_preserve_default(vendor: u32, device: u32) -> bool {
    vendor != 0x10de || device >= 0x2b80
}

/// Translated shaders declare maximal reconvergence (`exec::module_code`); set when the device is
/// made.
static MAX_RECONV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether every translated shader declares maximal reconvergence (`MaximallyReconvergesKHR`,
/// VK_KHR_shader_maximal_reconvergence; `exec::module_code` adds it to each module). The
/// translation assumes that a subgroup's invocations come back together at each structured merge
/// (cross-lane reads after a lane-guarded region, the dispatcher's rounds: RENDERER.md "Wave
/// semantics"), which Vulkan promises only to a shader declaring it; NVIDIA's compiler did so on
/// the RTX 40 series without it. On an RTX 5090 and an RTX 5080 (driver 621.24) players saw the
/// tunnel at the game's start black with what looked like exploding geometry, in 0.1.1 too, with
/// every compute program in the portable translation (issue #4, 2026-11-24): on the designs the
/// subgroup modes were not checked on (`subgroup_modes_checked`) shaders declare it since then.
/// `DS_MAX_RECONV=1` / `0` forces it on or off where the device offers it.
pub fn max_reconvergence() -> bool {
    MAX_RECONV.load(std::sync::atomic::Ordering::Relaxed)
}

/// Translated shaders declare SignedZeroInfNanPreserve (`exec::module_code`); set when the device
/// is made, with the widths the device allows (bit 0: 16, bit 1: 32, bit 2: 64).
static NAN_PRESERVE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether every translated shader declares `SignedZeroInfNanPreserve` (SPV_KHR_float_controls).
/// Without it Vulkan lets a compiler assume that no value is a NaN or an infinity and ignore the
/// sign of zero, so it may fold an `OpIsNan` or `OpIsInf` to false. The translation relies on
/// them where the console's hardware does: a vertex whose position has a NaN or infinite
/// component is moved outside the clip volume so that its primitive is dropped, as the console's
/// clipper drops it (the title's instancing gives unused instance slots NaN positions;
/// `translate::mem`'s position export), and v_cmp_class tests classes. On the RTX 40 series the
/// tests held without the declaration; before that guard the RTX 4090 drew those primitives as
/// streaks across the tutorial's opening tunnel, and on the RTX 50 series (driver 621.24 and
/// 621.35) players see that tunnel black with "vertex explosions" (issues #4 and #6, 2026-11-24):
/// on Blackwell, AMD and Intel (`nan_preserve_default`) shaders declare it
/// since then, for the widths the device allows (the RTX 4090 and 5090 report all three).
/// `DS_NAN_PRESERVE=1` / `0` forces it on or off where the device allows it.
pub fn nan_preserve() -> bool {
    NAN_PRESERVE.load(std::sync::atomic::Ordering::Relaxed) & 2 != 0
}

/// The widths `nan_preserve` declares (bit 0: 16, bit 1: 32, bit 2: 64).
pub fn nan_preserve_widths() -> u32 {
    NAN_PRESERVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// `DS_MAX_RECONV`: Some(on) when set to 1 or 0.
fn max_reconv_forced() -> Option<bool> {
    match std::env::var("DS_MAX_RECONV").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

thread_local! {
    /// What this thread is creating (`creating`), for the validation
    /// layer's messages.
    static CREATING: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Notes what this thread creates until the guard drops, for the validation
/// layer's messages (only under it).
pub fn creating(what: impl FnOnce() -> String) -> Option<CreatingGuard> {
    try_ctx().filter(|c| c.validation)?;
    CREATING.with(|c| *c.borrow_mut() = what());
    Some(CreatingGuard)
}

pub struct CreatingGuard;

impl Drop for CreatingGuard {
    fn drop(&mut self) {
        CREATING.with(|c| c.borrow_mut().clear());
    }
}

/// Names an object for the validation layer's messages (only under it).
pub fn name_object<H: vk::Handle>(handle: H, name: &str) {
    if let Some(n) = ctx().names.as_ref() {
        if let Ok(name) = std::ffi::CString::new(name) {
            let _ = unsafe { n.set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT::default().object_handle(handle).object_name(&name)) };
        }
    }
}

/// The device takes the mirror-once sampler address mode (`samplerMirrorClampToEdge`, enabled).
static MIRROR_CLAMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The device waits for presents (`DS_PRESENT_STATS=1`: VK_KHR_present_id and present_wait).
pub static PRESENT_WAIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether samplers may use `MIRROR_CLAMP_TO_EDGE` (`Images::sampler`).
pub fn mirror_clamp() -> bool {
    MIRROR_CLAMP.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn ctx() -> &'static Ctx {
    CTX.get().expect("Vulkan not initialized")
}

pub fn try_ctx() -> Option<&'static Ctx> {
    CTX.get()
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    // (The layer reports a module's SPIR-V while creating it, on the
    // creating thread: what that thread is creating, `creating`.)
    let what = CREATING.with(|c| c.borrow().clone());
    let msg = if what.is_empty() { msg.into_owned() } else { format!("{msg} [while creating {what}]") };
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        crate::error!("vulkan: {msg}");
    } else {
        crate::warn!("vulkan: {msg}");
    }
    vk::FALSE
}

/// Vulkan initialization step reached (an init watchdog names it: two boots
/// hung before initialization logged anything, the queues waiting for it).
static INIT_STAGE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const INIT_STAGES: [&str; 6] = ["loading the Vulkan loader", "creating the instance", "enumerating devices", "creating the device", "finishing", "done"];

fn init_stage(k: u8) {
    INIT_STAGE.store(k, std::sync::atomic::Ordering::SeqCst);
}

pub fn init() -> Result<(), String> {
    if let Some(n) = std::env::var("DS_LOOP_LIMIT").ok().and_then(|v| v.parse::<u32>().ok()).filter(|&n| n > 0) {
        rdna::translate::set_loop_limit(n);
        crate::info!("gpu: loop watchdog limit {n} blocks per invocation (DS_LOOP_LIMIT)");
    }
    if CTX.get().is_some() {
        return Ok(());
    }
    // Reports a driver call that does not return (every queue waits for init).
    let _ = std::thread::Builder::new().name("vk-init-watchdog".into()).spawn(|| {
        for s in 1..=12 {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let k = INIT_STAGE.load(std::sync::atomic::Ordering::SeqCst) as usize;
            if k >= INIT_STAGES.len() - 1 {
                return;
            }
            crate::warn!("Vulkan initialization still {} after {} s", INIT_STAGES[k], 10 * s);
        }
    });
    let r = init_inner();
    init_stage(INIT_STAGES.len() as u8 - 1);
    r
}

/// The validation layer of a Vulkan SDK copied into the project
/// (local-data/research/vulkan-sdk/<version>/Bin: the LunarG installer's
/// `copy_only=1`, nothing registered with the system), the newest version.
fn local_layer_dir() -> Option<std::path::PathBuf> {
    let data = crate::run_dir()?.parent()?.parent()?.to_path_buf();
    let mut dirs: Vec<_> = std::fs::read_dir(data.join("research").join("vulkan-sdk")).ok()?.flatten().map(|e| e.path().join("Bin")).filter(|p| p.join("VkLayer_khronos_validation.json").is_file()).collect();
    dirs.sort();
    dirs.pop()
}

fn init_inner() -> Result<(), String> {
    unsafe {
        init_stage(0);
        // DS_VK_VALIDATION without a layer path of its own: the project's
        // SDK copy (`local_layer_dir`), for the loader to find (set before it
        // loads; this diagnosis only).
        if validation_requested(std::env::var_os("DS_VK_VALIDATION").as_deref()) && std::env::var_os("VK_ADD_LAYER_PATH").is_none() {
            if let Some(dir) = local_layer_dir() {
                crate::info!("Vulkan validation layer from {}", dir.display());
                std::env::set_var("VK_ADD_LAYER_PATH", &dir);
            }
        }
        let entry = ash::Entry::load().map_err(|e| format!("loading Vulkan: {e}"))?;
        init_stage(1);
        let app_name = c"dsport";
        let app = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let layers_avail = entry.enumerate_instance_layer_properties().unwrap_or_default();
        let want_validation = validation_requested(std::env::var_os("DS_VK_VALIDATION").as_deref());
        let validation_name = c"VK_LAYER_KHRONOS_validation";
        let has_validation = layers_avail.iter().any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
        let validation = want_validation && has_validation;
        if want_validation && !has_validation {
            crate::warn!("DS_VK_VALIDATION set but the Khronos validation layer is not installed");
        }
        let layers: Vec<*const c_char> = if validation { vec![validation_name.as_ptr()] } else { vec![] };
        let surface_extensions = window::instance_extensions()?;
        let mut exts: Vec<_> = surface_extensions.iter().map(|s| s.as_ptr()).collect();
        let inst_exts = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
        let inst_has = |n: &CStr| inst_exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
        // Display modes and HDR (`present`): surface queries with chained structures (exclusive
        // fullscreen needs them) and the HDR10 color spaces, where the loader offers them.
        let surface_caps2 = inst_has(ash::khr::get_surface_capabilities2::NAME);
        if surface_caps2 {
            exts.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());
        }
        let colorspaces = cfg!(windows) && inst_has(ash::ext::swapchain_colorspace::NAME);
        if colorspaces {
            exts.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
        }
        // Command labels for GPU Trace (DS_GPU_LABELS=1).
        let want_labels = std::env::var("DS_GPU_LABELS").ok().as_deref() == Some("1");
        let has_debug_utils = inst_has(ash::ext::debug_utils::NAME);
        let labels_on = want_labels && has_debug_utils;
        if want_labels && !has_debug_utils {
            crate::warn!("DS_GPU_LABELS=1 but VK_EXT_debug_utils is unavailable");
        }
        if validation || labels_on {
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        // The upscalers' instance extensions (`super::upscale`: XeSS's, Streamline's for DLSS).
        let up_inst = crate::gpu::upscale::instance_extensions(&inst_exts);
        for e in &up_inst {
            if !exts.iter().any(|&x| CStr::from_ptr(x) == e.as_c_str()) {
                exts.push(e.as_ptr());
            }
        }
        let ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&exts);
        let instance = entry.create_instance(&ici, None).map_err(|e| format!("vkCreateInstance: {e}"))?;
        init_stage(2);
        if validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let mci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .pfn_user_callback(Some(debug_cb));
            let _ = du.create_debug_utils_messenger(&mci, None);
        }
        // Pick a device the port can use: Vulkan 1.3, a graphics+compute queue and the extensions
        // it always enables; a native driver before a translation layer's device (Microsoft's
        // D3D12 one, installed with Windows' compatibility pack, lists every card again, without
        // push descriptors), then by kind: discrete, integrated, virtual, the CPU's; among
        // equals the last discrete one, else the first (the old rule; a CPU driver such as
        // lavapipe passes every check, the second review of 2026-11-25). (Until 2026-11-25 the
        // last discrete device was taken whatever it offered: a player's RTX 3060 Ti failed with
        // "vkCreateDevice: Extension specified does not exist", likely such a device.) None
        // usable: the old rule, and its error.
        let pdevs = instance.enumerate_physical_devices().map_err(|e| format!("enumerate devices: {e}"))?;
        let mut chosen: Option<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> = None;
        let mut chosen_rank = i32::MIN;
        let mut fallback: Option<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> = None;
        let mut seen = Vec::new();
        for &pd in &pdevs {
            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            let mut dp = vk::PhysicalDeviceDriverProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut dp);
            instance.get_physical_device_properties2(pd, &mut p2);
            let driver = CStr::from_ptr(dp.driver_name.as_ptr()).to_string_lossy().into_owned();
            let layered = matches!(dp.driver_id, vk::DriverId::MESA_DOZEN | vk::DriverId::MESA_VENUS | vk::DriverId::MOLTENVK);
            let qfs = instance.get_physical_device_queue_family_properties(pd);
            let qf = qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE));
            let exts = instance.enumerate_device_extension_properties(pd).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            let mut lacks: Vec<String> = Vec::new();
            if vk::api_version_major(props.api_version) == 1 && vk::api_version_minor(props.api_version) < 3 {
                lacks.push(format!("Vulkan 1.3 (it offers 1.{})", vk::api_version_minor(props.api_version)));
            }
            if qf.is_none() {
                lacks.push("a graphics and compute queue".into());
            }
            for n in [ash::khr::swapchain::NAME, ash::khr::push_descriptor::NAME] {
                if !has(n) {
                    lacks.push(n.to_string_lossy().into_owned());
                }
            }
            let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            seen.push(format!(
                "{name} ({:?}, driver {driver}{}{})",
                props.device_type,
                if layered { ", a translation layer" } else { "" },
                if lacks.is_empty() { String::new() } else { format!("; lacks {}", lacks.join(", ")) }
            ));
            if let Some(qf) = qf {
                if fallback.is_none() || discrete {
                    fallback = Some((pd, qf as u32, props));
                }
                if lacks.is_empty() {
                    let kind = match props.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                        _ => 0,
                    };
                    // Preserve Linux's NVIDIA preference among usable native devices.
                    let rank = if layered { 0 } else { 20 } + kind * 2
                        + i32::from(cfg!(target_os = "linux") && props.vendor_id == 0x10de);
                    if rank > chosen_rank || (rank == chosen_rank && discrete) {
                        chosen_rank = rank;
                        chosen = Some((pd, qf as u32, props));
                    }
                }
            }
        }
        // (Every card Vulkan lists, for a player's report: a laptop with two graphics chips.)
        crate::info!(
            "Vulkan devices: {} (taken: a usable one, a native driver before a translation layer, then discrete, integrated, virtual, the CPU's (Linux prefers NVIDIA within a kind), among equals the last discrete one, else the first; none usable: the last discrete one, else the first)",
            seen.join(", ")
        );
        let (pdev, qf, props) = chosen.or(fallback).ok_or("no Vulkan device with graphics+compute")?;
        let device_name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
        // (The design decides the translation's subgroup modes: `shaders::plain_modes`.)
        let checked = modes_checked_on(props.vendor_id, props.device_id);
        SUBGROUP_MODES_CHECKED.store(checked, std::sync::atomic::Ordering::Relaxed);
        crate::info!(
            "Vulkan device ids: PCI {:04x}:{:04x}; the translation's subgroup modes {} (DS_PLAIN_MODES=1 or 0 overrides)",
            props.vendor_id,
            props.device_id,
            if checked { "on: a design they were checked on" } else { "off: not checked on this design, every compute program in the portable translation" }
        );
        // (The driver's version and pipeline cache id: players' logs are compared by them.)
        let driver_desc = {
            let v = props.driver_version;
            let ver = if props.vendor_id == 0x10de { format!("{}.{:02}", v >> 22, (v >> 14) & 0xff) } else { format!("{}.{}.{}", v >> 22, (v >> 12) & 0x3ff, v & 0xfff) };
            let uuid: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
            format!("driver {ver} ({v:#x}), pipeline cache {uuid}")
        };
        let prio = [1.0f32];
        // A transfer-only family for the copy queue (`DS_COPY_QUEUE=0`: none).
        let copy_family = if std::env::var("DS_COPY_QUEUE").ok().as_deref() != Some("0") {
            instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::TRANSFER) && !q.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
                .map(|f| f as u32)
        } else {
            None
        };
        // A compute-only family (ordered async compute).
        let compute_family = if async_compute_boot() {
            instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE) && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|f| f as u32)
        } else {
            None
        };
        // (Two queues of the graphics family when it has them: the second for the hidden
        // present, `Ctx::present_queue`.)
        let prio2 = [1.0f32, 0.5];
        // (`DS_PRESENT_QUEUE=0`: one queue, as on a device with one in that family: a test.)
        let gfx_queues = if std::env::var("DS_PRESENT_QUEUE").ok().as_deref() == Some("0") { 1 } else { instance.get_physical_device_queue_family_properties(pdev).get(qf as usize).map_or(1, |q| q.queue_count) };
        let mut qci = vec![vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(if gfx_queues >= 2 { &prio2[..] } else { &prio[..] })];
        if let Some(f) = copy_family {
            qci.push(vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio));
        }
        if let Some(f) = compute_family {
            qci.push(vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio));
        }
        let host_import = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::external_memory_host::NAME);
        let dev_exts = instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
        let dev_has = |n: &CStr| dev_exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
        let mut dexts = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::push_descriptor::NAME.as_ptr()];
        if host_import {
            dexts.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
        // RADV refuses a host pointer over shared memory but accepts a dma-buf
        // built from the section's memfd. Only Linux, only when both extensions exist.
        let dma_ok = cfg!(target_os = "linux")
            && dev_has(ash::khr::external_memory_fd::NAME)
            && dev_has(ash::ext::external_memory_dma_buf::NAME);
        if dma_ok {
            dexts.push(ash::khr::external_memory_fd::NAME.as_ptr());
            dexts.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
        }
        // Exclusive fullscreen under the application's control, and borderless windows the
        // driver may not turn into exclusive mode (`present`); HDR metadata for HDR10 output.
        let has_fse = cfg!(windows) && surface_caps2 && dev_has(ash::ext::full_screen_exclusive::NAME) && std::env::var("DS_NO_FSE").is_err();
        if has_fse {
            dexts.push(ash::ext::full_screen_exclusive::NAME.as_ptr());
        }
        // On-screen frame times (`DS_PRESENT_STATS=1`, diagnosis): each present's id waited for
        // until it is shown (VK_KHR_present_id, VK_KHR_present_wait; `present::stats_line`).
        let has_pw = std::env::var("DS_PRESENT_STATS").ok().as_deref() == Some("1") && dev_has(ash::khr::present_id::NAME) && dev_has(ash::khr::present_wait::NAME) && {
            let mut a = vk::PhysicalDevicePresentIdFeaturesKHR::default();
            let mut b = vk::PhysicalDevicePresentWaitFeaturesKHR::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut a).push_next(&mut b);
            instance.get_physical_device_features2(pdev, &mut f2);
            a.present_id != 0 && b.present_wait != 0
        };
        if has_pw {
            dexts.push(ash::khr::present_id::NAME.as_ptr());
            dexts.push(ash::khr::present_wait::NAME.as_ptr());
        }
        PRESENT_WAIT.store(has_pw, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("DS_PRESENT_STATS").ok().as_deref() == Some("1") {
            crate::info!("present waits (DS_PRESENT_STATS=1): {}", if has_pw { "on" } else { "the device has no present id or present wait" });
        }
        let mut pid_f = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        let mut pw_f = vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
        let has_hdr_meta = colorspaces && dev_has(ash::ext::hdr_metadata::NAME);
        if has_hdr_meta {
            dexts.push(ash::ext::hdr_metadata::NAME.as_ptr());
        }
        // The upscalers' device extensions and features (`super::upscale`).
        let up_dev = crate::gpu::upscale::device_needs(&instance, pdev);
        for e in &up_dev.extensions {
            if !dexts.iter().any(|&x| CStr::from_ptr(x) == e.as_c_str()) {
                dexts.push(e.as_ptr());
            }
        }
        let memory_budget = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::memory_budget::NAME);
        if memory_budget {
            dexts.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        let calibrated = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::calibrated_timestamps::NAME)
            && ash::ext::calibrated_timestamps::Instance::new(&entry, &instance)
                .get_physical_device_calibrateable_time_domains(pdev)
                .is_ok_and(|d| d.contains(&vk::TimeDomainEXT::DEVICE) && d.contains(&window::time_domain()));
        if calibrated {
            dexts.push(ash::ext::calibrated_timestamps::NAME.as_ptr());
        }
        // Device faults: the faulting addresses and the driver's description
        // after a device loss.
        let has_fault_ext = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::device_fault::NAME);
        let mut fault_supported = vk::PhysicalDeviceFaultFeaturesEXT::default();
        if has_fault_ext {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let has_fault = has_fault_ext && fault_supported.device_fault != 0;
        let mut fault_f = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true).device_fault_vendor_binary(fault_supported.device_fault_vendor_binary != 0);
        if has_fault {
            dexts.push(ash::ext::device_fault::NAME.as_ptr());
        }
        let want_ckpt = std::env::var("DS_GPU_CHECKPOINTS").ok().as_deref() == Some("1");
        let has_ckpt = want_ckpt
            && instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::nv::device_diagnostic_checkpoints::NAME);
        if has_ckpt {
            dexts.push(ash::nv::device_diagnostic_checkpoints::NAME.as_ptr());
        } else if want_ckpt {
            crate::warn!("DS_GPU_CHECKPOINTS=1 but VK_NV_device_diagnostic_checkpoints is unavailable");
        }
        let want_stats = std::env::var("DS_PIPELINE_STATS").ok().as_deref() == Some("1");
        let has_stats = want_stats
            && instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::pipeline_executable_properties::NAME);
        if has_stats {
            dexts.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        } else if want_stats {
            crate::warn!("DS_PIPELINE_STATS=1 but VK_KHR_pipeline_executable_properties is unavailable");
        }
        let mut stats_f = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        // Out-of-range buffer/image accesses must not fault: GCN bounds-checks
        // buffer accesses against the descriptor (loads return 0, stores are
        // dropped), and titles rely on it. robustness2 gives exactly that.
        let has_robust2 = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::robustness2::NAME);
        let mut r2_supported = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        if has_robust2 {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut r2_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let mut r2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(r2_supported.robust_buffer_access2 != 0)
            .robust_image_access2(r2_supported.robust_image_access2 != 0)
            .null_descriptor(r2_supported.null_descriptor != 0);
        if has_robust2 {
            dexts.push(ash::ext::robustness2::NAME.as_ptr());
        }
        // Fragment barycentrics: the title interpolates some attributes itself
        // from each vertex's value (v_interp_mov of P10/P20).
        let has_bary_ext = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::fragment_shader_barycentric::NAME);
        let mut bary_supported = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
        if has_bary_ext {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut bary_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let barycentric = has_bary_ext && bary_supported.fragment_shader_barycentric != 0 && gpu_compat() != Some("intel");
        // The title interpolates packed 16-bit attributes itself from each vertex's value and
        // the barycentric registers: without the extension a geometry shader hands those pixel
        // stages each vertex's values and the barycentrics (`DS_BARY_GS=0`: none, every pixel
        // gets vertex 0's value and surfaces look faceted, FINDINGS.md 2026-11-11).
        let bary_gs = !barycentric && instance.get_physical_device_features(pdev).geometry_shader != 0 && std::env::var("DS_BARY_GS").ok().as_deref() != Some("0");
        if bary_gs {
            crate::info!("Vulkan: no fragment shader barycentrics (VK_KHR_fragment_shader_barycentric): pixel shaders that interpolate their own attributes get them from a geometry shader");
        } else if !barycentric {
            crate::warn!("Vulkan: no fragment shader barycentrics (VK_KHR_fragment_shader_barycentric) and no geometry shaders: surfaces whose shaders interpolate their own attributes will look faceted on this device");
        }
        // Partitioned subgroup operations (write marks ORed per map word:
        // `rdna::translate::Spec::marks_part`).
        let partitioned = gpu_compat().is_none()
            && instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::nv::shader_subgroup_partitioned::NAME);
        if partitioned {
            dexts.push(ash::nv::shader_subgroup_partitioned::NAME.as_ptr());
        }
        let mut bary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default().fragment_shader_barycentric(true);
        if barycentric {
            dexts.push(ash::khr::fragment_shader_barycentric::NAME.as_ptr());
        }
        // Maximal reconvergence (`max_reconvergence`): on the designs the subgroup modes were not
        // checked on, or as `DS_MAX_RECONV` says, where the device offers it (with the subgroup
        // uniform control flow extension it requires). Not enabled elsewhere: the device the
        // checked designs' stored pipelines were made on stays as it was.
        let mr_wanted = max_reconv_forced().unwrap_or(!checked);
        let mr_offered = {
            let exts = instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            if has(ash::khr::shader_maximal_reconvergence::NAME) && has(ash::khr::shader_subgroup_uniform_control_flow::NAME) {
                let mut mr = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
                let mut ucf = vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default();
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut mr).push_next(&mut ucf);
                instance.get_physical_device_features2(pdev, &mut f2);
                mr.shader_maximal_reconvergence != 0 && ucf.shader_subgroup_uniform_control_flow != 0
            } else {
                false
            }
        };
        let max_reconv = mr_wanted && mr_offered;
        if max_reconv {
            dexts.push(ash::khr::shader_subgroup_uniform_control_flow::NAME.as_ptr());
            dexts.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
        }
        MAX_RECONV.store(max_reconv, std::sync::atomic::Ordering::Relaxed);
        crate::info!("effective translation policy: source={} portable={} maximal_reconvergence={} per_lane={} (DS_PLAIN_MODES={:?}, DS_MAX_RECONV={:?}); pipeline binaries verify SPIR-V plus reconvergence fingerprint",
            rdna::SOURCE_HASH, super::shaders::plain_modes(), max_reconv, super::tune::per_lane(),
            std::env::var("DS_PLAIN_MODES").ok(), std::env::var("DS_MAX_RECONV").ok());
        crate::info!(
            "Vulkan: maximal reconvergence {} (offered {mr_offered}, wanted {mr_wanted}: {}; DS_MAX_RECONV=1 or 0 overrides)",
            if max_reconv { "on: every translated shader declares it" } else { "off" },
            match max_reconv_forced() {
                Some(_) => "DS_MAX_RECONV",
                None if checked => "a design the translation was checked on",
                None => "a design the translation was not checked on",
            }
        );
        let mut mr_f = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default().shader_maximal_reconvergence(true);
        let mut ucf_f = vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default().shader_subgroup_uniform_control_flow(true);
        // Subgroups: the translation's cross-lane code (ballots, lane reads, wave halves) is
        // written for subgroups of 32 lanes, NVIDIA's only size. Where the device's default
        // differs (AMD 64, Intel 8 to 32), the stages it lets a pipeline give a size get 32
        // (subgroupSizeControl, Vulkan 1.3); `DS_SUBGROUP32=1` does so on any device (a test
        // of the path), `=0` never.
        let (subgroup_size, subgroup32, sg_note) = {
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p13);
            instance.get_physical_device_properties2(pdev, &mut p2);
            let mut f13s = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13s);
            instance.get_physical_device_features2(pdev, &mut f2);
            let control = f13s.subgroup_size_control != 0 && p13.min_subgroup_size <= 32 && p13.max_subgroup_size >= 32;
            let wanted = match std::env::var("DS_SUBGROUP32").ok().as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => p11.subgroup_size != 32 || gpu_compat().is_some(),
            };
            // AMD RDNA2 and Intel let compute alone be given a size (RDNA3 and 4 fragment too).
            let allowed = if gpu_compat().is_some() { p13.required_subgroup_size_stages & vk::ShaderStageFlags::COMPUTE } else { p13.required_subgroup_size_stages };
            let stages = if wanted && control { allowed } else { vk::ShaderStageFlags::empty() };
            let note = format!(
                "Vulkan subgroups: default {} lanes (sizes {}-{}; a size can be required for {:?}{}); 32 required for {:?}",
                p11.subgroup_size,
                p13.min_subgroup_size,
                p13.max_subgroup_size,
                p13.required_subgroup_size_stages,
                if f13s.subgroup_size_control != 0 { "" } else { ", subgroupSizeControl unsupported" },
                stages
            );
            (p11.subgroup_size, stages, note)
        };
        crate::info!("{sg_note}");
        // (What the driver lets a module declare about denormals, signed zeros, infinities, NaNs and
        // rounding: translated modules declare none of it, so the driver's defaults decide; players'
        // logs say what their design offers.)
        {
            let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p12);
            instance.get_physical_device_properties2(pdev, &mut p2);
            let b = |x: vk::Bool32| u32::from(x != 0);
            let widths = b(p12.shader_signed_zero_inf_nan_preserve_float16) | b(p12.shader_signed_zero_inf_nan_preserve_float32) << 1 | b(p12.shader_signed_zero_inf_nan_preserve_float64) << 2;
            let forced = match std::env::var("DS_NAN_PRESERVE").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            };
            // Float preservation is independent of Linux's validated NVIDIA subgroup modes.
            let preserve_default = nan_preserve_default(props.vendor_id, props.device_id);
            let on = forced.unwrap_or(preserve_default) && widths & 2 != 0;
            NAN_PRESERVE.store(if on { widths } else { 0 }, std::sync::atomic::Ordering::Relaxed);
            crate::info!(
                "Vulkan: NaN, infinity and signed-zero preservation {} ({}; DS_NAN_PRESERVE=1 or 0 overrides)",
                if on { "declared by every translated shader" } else { "not declared" },
                match forced {
                    Some(_) => "DS_NAN_PRESERVE",
                    None if !preserve_default => "a design whose float behavior was checked",
                    None => "explicit float preservation for this design",
                }
            );
            crate::info!(
                "Vulkan float controls: denormal modes {:?}, rounding modes {:?}; f16/f32/f64: signed zero, inf and NaN preserve {}{}{}, denormal preserve {}{}{}, flush to zero {}{}{}, RTE {}{}{}, RTZ {}{}{}",
                p12.denorm_behavior_independence,
                p12.rounding_mode_independence,
                b(p12.shader_signed_zero_inf_nan_preserve_float16),
                b(p12.shader_signed_zero_inf_nan_preserve_float32),
                b(p12.shader_signed_zero_inf_nan_preserve_float64),
                b(p12.shader_denorm_preserve_float16),
                b(p12.shader_denorm_preserve_float32),
                b(p12.shader_denorm_preserve_float64),
                b(p12.shader_denorm_flush_to_zero_float16),
                b(p12.shader_denorm_flush_to_zero_float32),
                b(p12.shader_denorm_flush_to_zero_float64),
                b(p12.shader_rounding_mode_rte_float16),
                b(p12.shader_rounding_mode_rte_float32),
                b(p12.shader_rounding_mode_rte_float64),
                b(p12.shader_rounding_mode_rtz_float16),
                b(p12.shader_rounding_mode_rtz_float32),
                b(p12.shader_rounding_mode_rtz_float64),
            );
        }
        if subgroup_size != 32 {
            let left = (vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::GEOMETRY | vk::ShaderStageFlags::FRAGMENT) & !subgroup32;
            if !left.is_empty() {
                crate::warn!("Vulkan subgroups: {left:?} programs run in subgroups of the device's choice ({subgroup_size} by default): cross-lane operations written for 32 lanes there approximate the title's (untested off NVIDIA)");
            }
        }
        let first_instance = instance.get_physical_device_features(pdev).draw_indirect_first_instance != 0;
        // Sparse buffers for the residency mirror (`DS_RESIDENT=1`): the
        // device and the main queue's family must support them.
        let sparse_buffers = (super::resident::wanted() || super::fe2::wanted()) && {
            let f = instance.get_physical_device_features(pdev);
            let fam = instance.get_physical_device_queue_family_properties(pdev)[qf as usize].queue_flags;
            f.sparse_binding != 0 && f.sparse_residency_buffer != 0 && fam.contains(vk::QueueFlags::SPARSE_BINDING)
        };
        if (super::resident::wanted() || super::fe2::wanted()) && !sparse_buffers {
            crate::warn!("front end 2 (the default), DS_RESIDENT or DS_RGM wanted but the device or the main queue family has no sparse buffers");
        }
        // Storage images bound one view per mip level are indexed per
        // invocation (image_store_mip / image_load_mip).
        let core = instance.get_physical_device_features(pdev);
        let storage_dyn = core.shader_storage_image_array_dynamic_indexing != 0;
        {
            let missing: Vec<&str> = [
                (core.geometry_shader, "geometryShader"),
                (core.independent_blend, "independentBlend"),
                (core.dual_src_blend, "dualSrcBlend"),
                (core.logic_op, "logicOp"),
                (core.fill_mode_non_solid, "fillModeNonSolid"),
                (core.depth_clamp, "depthClamp"),
                (core.depth_bias_clamp, "depthBiasClamp"),
                (core.sampler_anisotropy, "samplerAnisotropy"),
                (core.texture_compression_bc, "textureCompressionBC"),
                (core.image_cube_array, "imageCubeArray"),
                (core.shader_int64, "shaderInt64"),
                (core.shader_storage_image_write_without_format, "shaderStorageImageWriteWithoutFormat"),
                (core.shader_storage_image_read_without_format, "shaderStorageImageReadWithoutFormat"),
                (core.vertex_pipeline_stores_and_atomics, "vertexPipelineStoresAndAtomics"),
                (core.fragment_stores_and_atomics, "fragmentStoresAndAtomics"),
                (core.multi_draw_indirect, "multiDrawIndirect"),
                (core.sample_rate_shading, "sampleRateShading"),
            ]
            .into_iter()
            .filter(|(v, _)| *v == 0)
            .map(|(_, n)| n)
            .collect();
            if !missing.is_empty() {
                crate::warn!("Vulkan: the device lacks {} (the programs or states that need them fail or draw wrong; untested)", missing.join(", "));
            }
        }
        // Gathers with offsets declare ImageGatherExtended (the validation
        // layer reported every such module without it).
        let mut features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_array_dynamic_indexing(storage_dyn)
            .shader_image_gather_extended(core.shader_image_gather_extended != 0)
            .draw_indirect_first_instance(first_instance)
            .robust_buffer_access(true)
            .geometry_shader(core.geometry_shader != 0)
            .independent_blend(core.independent_blend != 0)
            .dual_src_blend(core.dual_src_blend != 0)
            .logic_op(core.logic_op != 0)
            .fill_mode_non_solid(core.fill_mode_non_solid != 0)
            .depth_clamp(core.depth_clamp != 0)
            .depth_bias_clamp(core.depth_bias_clamp != 0)
            .sampler_anisotropy(core.sampler_anisotropy != 0)
            .texture_compression_bc(core.texture_compression_bc != 0)
            .image_cube_array(core.image_cube_array != 0)
            .shader_int64(core.shader_int64 != 0)
            .shader_storage_image_write_without_format(core.shader_storage_image_write_without_format != 0)
            .shader_storage_image_read_without_format(core.shader_storage_image_read_without_format != 0)
            .vertex_pipeline_stores_and_atomics(core.vertex_pipeline_stores_and_atomics != 0)
            .fragment_stores_and_atomics(core.fragment_stores_and_atomics != 0)
            .multi_draw_indirect(core.multi_draw_indirect != 0)
            .sample_rate_shading(core.sample_rate_shading != 0)
            .sparse_binding(sparse_buffers)
            .sparse_residency_buffer(sparse_buffers);
        // Image tables are indexed per invocation (non-uniform indexing).
        let mut f12_supported = vk::PhysicalDeviceVulkan12Features::default();
        let mut f11_supported = vk::PhysicalDeviceVulkan11Features::default();
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12_supported).push_next(&mut f11_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let draw_parameters = f11_supported.shader_draw_parameters != 0;
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(draw_parameters);
        let nonuniform = f12_supported.shader_sampled_image_array_non_uniform_indexing != 0;
        if !nonuniform {
            crate::warn!("Vulkan: shaderSampledImageArrayNonUniformIndexing unsupported (image tables index uniformly)");
        }
        let storage_nonuniform = f12_supported.shader_storage_image_array_non_uniform_indexing != 0;
        if !storage_nonuniform || !storage_dyn {
            crate::warn!("Vulkan: storage image arrays without non-uniform indexing (programs that select mip levels per lane may fail or write wrong levels)");
        }
        let device_addresses = f12_supported.buffer_device_address != 0;
        let draw_indirect_count = f12_supported.draw_indirect_count != 0;
        let host_query_reset = f12_supported.host_query_reset != 0;
        // (The title's samplers use the mirror-once address mode; without the feature they take
        // mirrored repeat, `mirror_clamp`. Not enabled before 2026-11-15: the validation layer
        // reported the samplers.)
        let mirror_clamp = f12_supported.sampler_mirror_clamp_to_edge != 0;
        MIRROR_CLAMP.store(mirror_clamp, std::sync::atomic::Ordering::Relaxed);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .sampler_mirror_clamp_to_edge(mirror_clamp)
            .buffer_device_address(device_addresses)
            .host_query_reset(host_query_reset)
            .draw_indirect_count(draw_indirect_count)
            .shader_sampled_image_array_non_uniform_indexing(nonuniform)
            .shader_storage_image_array_non_uniform_indexing(storage_nonuniform)
            .timeline_semaphore(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .shader_int8(true);
        // Private data slots (Vulkan 1.3's privateData) where the device has them: NVIDIA's DLSS
        // library makes one at its start (the validation layer reported the feature missing,
        // 2026-11-11).
        let private_data = {
            let mut f = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
            instance.get_physical_device_features2(pdev, &mut f2);
            f.private_data != 0
        };
        // (pipelineCreationCacheControl, required in 1.3: `pcache::pb_check` asks the driver to
        // serve a pipeline from cache data or fail.)
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true).maintenance4(true).subgroup_size_control(!subgroup32.is_empty()).private_data(private_data).pipeline_creation_cache_control(true);
        // Pipeline binaries (`pipeline_binaries`): VK_KHR_maintenance5 (for
        // the capture flag) and VK_KHR_pipeline_binary with its feature.
        let has_m5 = instance.enumerate_device_extension_properties(pdev).unwrap_or_default().iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::maintenance5::NAME);
        // (The store's mode file may turn them off for this card and driver: its driver refused
        // binaries another process made, `pcache::binaries_off_for`.)
        let cache_id: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
        crate::gpu::pcache::set_identity(&device_name, &cache_id);
        let store_off = crate::gpu::pcache::binaries_off_for(&device_name, &cache_id);
        if let Some(why) = &store_off {
            crate::info!("Vulkan pipeline binaries: off for this store's card and driver (binaries-mode.txt: {why}): the store keeps cache data, the driver's own cache on");
        }
        let pbins = pipeline_binaries() && store_off.is_none() && has_m5 && pbin::supported(&instance, pdev);
        if pipeline_binaries() && store_off.is_none() && !pbins {
            crate::warn!("Vulkan: pipeline binaries asked for but not available (maintenance5 {has_m5}): the pipeline store keeps cache data only");
        }
        if pbins {
            dexts.push(ash::khr::maintenance5::NAME.as_ptr());
            dexts.push(pbin::NAME.as_ptr());
        }
        let mut m5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true);
        let mut pb_feat = pbin::Features { s_type: pbin::ST_FEATURES, p_next: std::ptr::null_mut(), pipeline_binaries: 1 };
        let mut pb_cache = pbin::CacheControl { s_type: pbin::ST_CACHE_CONTROL, p_next: std::ptr::null(), disable_internal_cache: 1 };
        crate::gpu::upscale::apply_features(&mut features, &mut f11, &mut f12, &mut f13, &up_dev);
        #[cfg(target_os = "linux")]
        crate::gpu::framegen::apply_features(&core, &mut features);
        crate::info!("Vulkan enabled extensions: {}", dexts.iter().map(|&p| CStr::from_ptr(p).to_string_lossy()).collect::<Vec<_>>().join(", "));
        crate::info!("Vulkan enabled features: synchronization2={} timelineSemaphore={} dynamicRendering={} bufferDeviceAddress={} subgroupSizeControl={}; queues: {}",
            f13.synchronization2, f12.timeline_semaphore, f13.dynamic_rendering, f12.buffer_device_address, f13.subgroup_size_control,
            qci.iter().map(|q| format!("family {} count {}", q.queue_family_index, q.queue_count)).collect::<Vec<_>>().join(", "));
        crate::info!("Vulkan renderer paths: push descriptors; device-local images and FE2 sparse mirror; host imports feed mirror uploads. Descriptor buffer, host image copy and pageable device-local memory are not renderer paths on either platform.");
        let mut mutable = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default().mutable_descriptor_type(true);
        // Value-only compatibility identity, before push_next links these feature structs.
        // Driver UUID/version alone does not distinguish enabled robustness/subgroup features.
        let mut binary_extensions: Vec<_> = dexts.iter().map(|&p| CStr::from_ptr(p).to_string_lossy().into_owned()).collect();
        binary_extensions.sort();
        let binary_present_wait = has_pw;
        let binary_features = format!("{features:?}|{f11:?}|{f12:?}|{f13:?}|{}|{:?}|{barycentric}|{max_reconv}|{}|{pbins}|{binary_present_wait}|{has_fault}|{has_stats}",
            has_robust2, if has_robust2 { Some(r2) } else { None }, up_dev.mutable_descriptor);
        let binary_compatibility = format!("contract1|{binary_extensions:?}|{binary_features}");
        crate::info!("Vulkan pipeline compatibility: {binary_compatibility}");
        // Image views clamped to the T#'s MIN_LOD (`images`): the title's streaming sets it to
        // its finest resident mip, and the mips below stay unmapped (zero-filled here, black).
        // Enabled after the pipelines' compatibility identity on purpose: a view's own clamp,
        // nothing a pipeline or shader is made from (stored pipelines stay valid).
        let view_min_lod = std::env::var("DS_VIEW_MIN_LOD").ok().as_deref() != Some("0")
            && instance.enumerate_device_extension_properties(pdev).unwrap_or_default().iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::image_view_min_lod::NAME)
            && {
                let mut f = vk::PhysicalDeviceImageViewMinLodFeaturesEXT::default();
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
                instance.get_physical_device_features2(pdev, &mut f2);
                f.min_lod != 0
            };
        let mut vml = vk::PhysicalDeviceImageViewMinLodFeaturesEXT::default().min_lod(true);
        if view_min_lod {
            dexts.push(ash::ext::image_view_min_lod::NAME.as_ptr());
        }
        crate::info!("Vulkan image views clamped to the T# MIN_LOD: {view_min_lod}");
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dexts)
            .enabled_features(&features)
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if up_dev.mutable_descriptor {
            dci = dci.push_next(&mut mutable);
        }
        if has_robust2 {
            dci = dci.push_next(&mut r2);
        }
        if has_pw {
            dci = dci.push_next(&mut pid_f).push_next(&mut pw_f);
        }
        if barycentric {
            dci = dci.push_next(&mut bary);
        }
        if max_reconv {
            dci = dci.push_next(&mut mr_f).push_next(&mut ucf_f);
        }
        if has_fault {
            dci = dci.push_next(&mut fault_f);
        }
        if view_min_lod {
            dci = dci.push_next(&mut vml);
        }
        if has_stats {
            dci = dci.push_next(&mut stats_f);
        }
        if pbins {
            dci = dci.push_next(&mut m5);
            pb_feat.p_next = dci.p_next as *mut std::ffi::c_void;
            dci.p_next = &pb_feat as *const pbin::Features as *const std::ffi::c_void;
            // (The driver's own cache on beside the binaries since 2026-11-21 night: a binary the
            // driver refuses is compiled again, from that cache when it holds the pipeline;
            // `DS_PB_NO_INTERNAL=1` turns it off for this device, as before.)
            if std::env::var("DS_PB_NO_INTERNAL").ok().as_deref() == Some("1") {
                pb_cache.p_next = dci.p_next;
                dci.p_next = &pb_cache as *const pbin::CacheControl as *const std::ffi::c_void;
                crate::info!("Vulkan pipeline binaries: the driver's own cache is off for this device (the store keeps the binaries)");
            } else {
                crate::info!("Vulkan pipeline binaries: the driver's own cache on beside them (a refused binary's compile served from it when it holds the pipeline)");
            }
        }
        init_stage(3);
        let device = instance.create_device(pdev, &dci, None).map_err(|e| format!("vkCreateDevice: {e}"))?;
        // The upscalers get the device (Streamline's DLSS starts on a thread of its own).
        crate::gpu::upscale::device_created(&instance, pdev, &device, qf, qci.iter().find(|q| q.queue_family_index == qf).map_or(1, |q| q.queue_count));
        // Its command functions go through deferred recording's (`defer`).
        let device = defer::install(device);
        init_stage(4);
        crate::info!(
            "Vulkan robustness: robustBufferAccess, robustness2 {} (buffer {}, image {}, null descriptors {})",
            has_robust2,
            r2_supported.robust_buffer_access2 != 0,
            r2_supported.robust_image_access2 != 0,
            r2_supported.null_descriptor != 0
        );
        let queue = device.get_device_queue(qf, 0);
        let present_queue = (gfx_queues >= 2).then(|| Mutex::new(device.get_device_queue(qf, 1)));
        let timeline = || {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("timeline semaphore: {e}"))
        };
        let compute = match compute_family {
            Some(f) => Some(ComputeQueue { family: f, queue: Mutex::new(device.get_device_queue(f, 0)), batches: timeline()? }),
            None => None,
        };
        let g_progress = if compute.is_some() { timeline()? } else { vk::Semaphore::null() };
        if let Some(c) = &compute {
            crate::info!("Vulkan: compute queue in family {} (ordered async compute: the compute queues' batches beside the main queue)", c.family);
        } else if async_compute_boot() {
            crate::warn!("Vulkan: no compute-only queue family: ordered async compute off");
        }
        let mut families = vec![qf];
        for f in [copy_family, compute_family].into_iter().flatten() {
            if !families.contains(&f) {
                families.push(f);
            }
        }
        let copy = copy_family.and_then(|f| {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            let copied = device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).ok()?;
            Some(CopyQueue { family: f, queue: Mutex::new(device.get_device_queue(f, 0)), copied })
        });
        if let Some(c) = &copy {
            crate::info!("Vulkan: copy queue in family {} (batch-start copies beside the main queue)", c.family);
        }
        // Formats the title samples with depth comparison: which support it
        // (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT).
        {
            let mut have = Vec::new();
            for f in [vk::Format::R16_UNORM, vk::Format::R32_SFLOAT, vk::Format::R16_SFLOAT, vk::Format::D16_UNORM, vk::Format::D32_SFLOAT] {
                let mut p3 = vk::FormatProperties3::default();
                let mut p2 = vk::FormatProperties2::default().push_next(&mut p3);
                instance.get_physical_device_format_properties2(pdev, f, &mut p2);
                let opt = p3.optimal_tiling_features.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_DEPTH_COMPARISON);
                have.push(format!("{f:?} {opt}"));
            }
            crate::info!("Vulkan depth-comparison sampling support: {}", have.join(", "));
        }
        let mem_props = instance.get_physical_device_memory_properties(pdev);
        let max_push = {
            let mut pd = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pd);
            instance.get_physical_device_properties2(pdev, &mut p2);
            pd.max_push_descriptors
        };
        let batches = {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("batch timeline semaphore: {e}"))?
        };
        let cpu_done = {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("effects timeline semaphore: {e}"))?
        };
        let pbin = if pbins { pbin::Fns::load(&instance, pdev, device.handle(), binary_compatibility.as_bytes()) } else { None };
        if pbins {
            match &pbin {
                Some(_) => crate::info!("Vulkan pipeline binaries: on"),
                None => crate::warn!("Vulkan pipeline binaries: the commands are missing (off)"),
            }
        }
        let ctx = Ctx {
            pbin,
            batches,
            cpu_done,
            surface_i: ash::khr::surface::Instance::new(&entry, &instance),
            #[cfg(windows)]
            win32_i: ash::khr::win32_surface::Instance::new(&entry, &instance),
            swapchain_d: ash::khr::swapchain::Device::new(&instance, &device),
            push_desc: ash::khr::push_descriptor::Device::new(&instance, &device),
            max_push,
            ext_host: host_import.then(|| ash::ext::external_memory_host::Device::new(&instance, &device)),
            dmabuf: dma_ok && std::path::Path::new("/dev/udmabuf").exists(),
            checkpoints: has_ckpt.then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device)),
            exec_props: has_stats.then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &device)),
            calibrated: calibrated.then(|| ash::ext::calibrated_timestamps::Device::new(&instance, &device)),
            fault: has_fault.then(|| ash::ext::device_fault::Device::new(&instance, &device)),
            fse: has_fse.then(|| ash::ext::full_screen_exclusive::Device::new(&instance, &device)),
            colorspaces,
            hdr_metadata: has_hdr_meta.then(|| ash::ext::hdr_metadata::Device::new(&instance, &device)),
            labels: labels_on.then(|| ash::ext::debug_utils::Device::new(&instance, &device)),
            names: validation.then(|| ash::ext::debug_utils::Device::new(&instance, &device)),
            device_addresses,
            robust2_null: r2_supported.robust_buffer_access2 != 0 && r2_supported.null_descriptor != 0,
            draw_indirect_count,
            sparse_buffers,
            draw_parameters: draw_parameters && first_instance,
            families,
            copy,
            compute,
            g_progress,
            g_count: AtomicU64::new(0),
            entry,
            instance,
            pdev,
            device,
            queue: Mutex::new(queue),
            queue_family: qf,
            present_queue,
            mem_props,
            limits: props.limits,
            device_name,
            validation,
            view_min_lod,
            host_query_reset,
            barycentric,
            bary_gs,
            partitioned,
            memory_budget,
            subgroup_size,
            subgroup32,
        };
        crate::info!("Vulkan device: {} (queue family {qf}, validation {}; {driver_desc})", ctx.device_name, validation);
        if let Some(v) = gpu_compat() {
            crate::info!("Vulkan: DS_GPU_COMPAT={v}: partitioned subgroups {}, fragment barycentrics {} (geometry shader instead {}), 32-lane subgroups required for {:?}", ctx.partitioned, ctx.barycentric, ctx.bary_gs, ctx.subgroup32);
        }
        {
            let exts = ctx.instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
            ctx.instance.get_physical_device_properties2(pdev, &mut p2);
            crate::info!(
                "Vulkan limits: maxStorageBufferRange {:#x}, minStorageBufferOffsetAlignment {}, external_memory_host {} (import alignment {:#x}), buffer_device_address {}, maxPushDescriptors {}, maxMemoryAllocationCount {}",
                ctx.limits.max_storage_buffer_range,
                ctx.limits.min_storage_buffer_offset_alignment,
                has(ash::ext::external_memory_host::NAME),
                host.min_imported_host_pointer_alignment,
                has(ash::khr::buffer_device_address::NAME),
                ctx.max_push,
                ctx.limits.max_memory_allocation_count
            );
        }
        CTX.set(ctx).ok();
        crate::info!("Vulkan memory heaps: {}", ctx_heaps());
        crate::info!("Vulkan memory types: {}", ctx_types());
        Ok(())
    }
}

impl Ctx {
    /// Whether pipelines give `stage` 32-lane subgroups (`subgroup32`): chain
    /// `subgroup32_info()` into its stage create info.
    pub fn wants_subgroup32(&self, stage: vk::ShaderStageFlags) -> bool {
        self.subgroup32.contains(stage)
    }

    /// vkCmdPushDescriptorSetKHR, into `cmd`'s list when this thread
    /// claimed it (`defer`).
    pub unsafe fn push_descriptors(&self, cmd: vk::CommandBuffer, bp: vk::PipelineBindPoint, layout: vk::PipelineLayout, set: u32, writes: &[vk::WriteDescriptorSet]) {
        if !defer::push_descriptors(cmd, bp, layout, set, writes) {
            defer::direct(cmd);
            unsafe { self.push_desc.cmd_push_descriptor_set(cmd, bp, layout, set, writes) }
        }
    }

    /// Each memory heap: size, flags and (VK_EXT_memory_budget) this
    /// process's usage against its budget, MiB.
    pub fn heap_report(&self) -> String {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default();
        if self.memory_budget {
            p2 = p2.push_next(&mut budget);
        }
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        let props = p2.memory_properties;
        (0..props.memory_heap_count as usize)
            .map(|i| {
                let h = props.memory_heaps[i];
                let local = if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) { "device-local" } else { "host" };
                if self.memory_budget {
                    format!("heap {i} ({local}) {} MiB: used {} of budget {}", h.size >> 20, budget.heap_usage[i] >> 20, budget.heap_budget[i] >> 20)
                } else {
                    format!("heap {i} ({local}) {} MiB", h.size >> 20)
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl Ctx {
    /// This process's usage of the largest device-local heap and its budget,
    /// bytes (None without VK_EXT_memory_budget).
    pub fn device_budget(&self) -> Option<(u64, u64)> {
        if !self.memory_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        let props = p2.memory_properties;
        let i = (0..props.memory_heap_count as usize).filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).max_by_key(|&i| props.memory_heaps[i].size)?;
        drop(p2);
        Some((budget.heap_usage[i].saturating_sub(sim_system_bytes()), budget.heap_budget[i].min(vram_cap()).min(sim_vram_limit().unwrap_or(u64::MAX))))
    }

    /// This process's usage of the largest heap that is not device-local (imported guest memory
    /// and the system-memory fallback are there) and its budget, the budget at most
    /// `DS_SIM_HOST_MIB`, bytes (None without VK_EXT_memory_budget or such a heap).
    pub fn host_budget(&self) -> Option<(u64, u64)> {
        if !self.memory_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        let props = p2.memory_properties;
        let i = (0..props.memory_heap_count as usize).filter(|&i| !props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).max_by_key(|&i| props.memory_heaps[i].size)?;
        drop(p2);
        Some((budget.heap_usage[i], budget.heap_budget[i].min(sim_host_limit().unwrap_or(u64::MAX))))
    }

    /// `DS_SIM_VRAM_MIB` (a test, `sim_vram_limit`): whether an allocation of `size` bytes in
    /// memory type `ty` fails as on a card whose room for this process ends at the limit: the
    /// type's heap is the largest device-local one and this process's usage of it plus the
    /// request would pass the limit. Counted in `SIM_VRAM_FAILS`.
    pub fn sim_alloc_fails(&self, ty: u32, size: u64) -> bool {
        let Some(limit) = sim_vram_limit() else { return false };
        if !self.memory_budget || ty >= self.mem_props.memory_type_count {
            return false;
        }
        let heap = self.mem_props.memory_types[ty as usize].heap_index as usize;
        let props = &self.mem_props;
        let largest = (0..props.memory_heap_count as usize).filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).max_by_key(|&i| props.memory_heaps[i].size);
        if largest != Some(heap) {
            return false;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        drop(p2);
        let fails = budget.heap_usage[heap].saturating_sub(sim_system_bytes()).saturating_add(size) > limit;
        if fails {
            SIM_VRAM_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fails
    }

    /// Whether memory type `ty` is in the largest device-local heap (the one the refusal window
    /// and `DS_SIM_VRAM_MIB` concern).
    fn in_main_heap(&self, ty: u32) -> bool {
        static HEAP: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        let props = &self.mem_props;
        let main = *HEAP.get_or_init(|| (0..props.memory_heap_count).filter(|&i| props.memory_heaps[i as usize].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).max_by_key(|&i| props.memory_heaps[i as usize].size));
        ty < props.memory_type_count && Some(props.memory_types[ty as usize].heap_index) == main
    }

    /// Whether an allocation of `size` bytes in memory type `ty` is refused before any driver
    /// call: inside the refusal window (`gate_refuses`) or as on a card with less room
    /// (`sim_alloc_fails`, noted as a failure). Other heaps' types are never refused here.
    pub fn refuse_alloc(&self, ty: u32, size: u64) -> bool {
        if !self.in_main_heap(ty) {
            return false;
        }
        if gate_refuses(size) {
            return true;
        }
        if self.sim_alloc_fails(ty, size) {
            self.note_alloc_fail(ty, size);
            return true;
        }
        false
    }

    /// A failed allocation of `size` bytes in memory type `ty`: in the largest device-local heap
    /// the refusal window begins (inside one, it takes the smaller size) and video memory counts
    /// as under pressure (`images::note_pressure`).
    pub fn note_alloc_fail(&self, ty: u32, size: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.in_main_heap(ty) {
            return;
        }
        let now = gate_clock_ms();
        let at = GATE_AT_MS.load(Relaxed);
        if at == 0 || now >= at + GATE_MS {
            GATE_SIZE.store(size, Relaxed);
            GATE_AT_MS.store(now, Relaxed);
        } else {
            GATE_SIZE.fetch_min(size, Relaxed);
        }
        GATE_STATS[1].fetch_add(1, Relaxed);
        super::images::note_pressure();
    }

    /// The size of the largest device-local heap, bytes (at most
    /// `vram_cap`).
    pub fn device_heap_size(&self) -> u64 {
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.pdev) };
        (0..props.memory_heap_count as usize).filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| props.memory_heaps[i].size).max().unwrap_or(0).min(vram_cap())
    }
}

/// `DS_IMPORT_LOG=1` (diagnosis): a memory allocation logged with its time, to line the
/// driver's submission stalls up with the allocations beside them.
pub fn note_alloc(what: &str, size: u64, t0: std::time::Instant) {
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("DS_IMPORT_LOG").ok().as_deref() == Some("1")) {
        crate::info!("gpu: allocated {what} ({} KiB) in {} us", size >> 10, t0.elapsed().as_micros());
    }
}

/// A main-queue call that took 2 ms or more (the queue's lock from `t0` to `t1`, the call after
/// it) as a hitch cause, with the pipeline creations running meanwhile (diagnosis: submissions
/// stalled while the driver compiles).
pub fn note_submit(what: &'static str, t0: std::time::Instant, t1: std::time::Instant) {
    let all = t0.elapsed();
    let lock = t1 - t0;
    if super::timeline::on() {
        super::timeline::split_back(&format!("Vulkan / {}", crate::log::thread_name()), lock, all.saturating_sub(lock), what);
    }
    if all.as_micros() < 2000 {
        return;
    }
    // (`DS_IMPORT_LOG=1`: each such call logged, to line up with the imports logged beside it.)
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("DS_IMPORT_LOG").ok().as_deref() == Some("1")) {
        crate::info!("gpu: {what} took {} us (queue lock {} us)", all.as_micros(), lock.as_micros());
    }
    crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::GpuWait, t0, || {
        format!(
            "{what} held up: queue lock {:.1} ms, call {:.1} ms; pipeline creations running {}",
            lock.as_secs_f64() * 1e3,
            (all - lock).as_secs_f64() * 1e3,
            crate::gpu::pcache::CREATING.load(Ordering::Relaxed)
        )
    });
}

/// `DS_GPU_COMPAT=amd|intel` (boot; a test of the fallbacks for other GPUs on this machine): the
/// device is used with the feature set current AMD (RDNA2) or Intel (Arc A-series) Windows
/// drivers have where NVIDIA can reproduce it (vulkan.gpuinfo.org reports of 2025-2026,
/// FINDINGS.md 2026-11-06 night): no partitioned subgroups (neither vendor has them), no
/// fragment barycentrics (Intel Arc A, Meteor Lake, Arrow Lake), 32-lane subgroups required in
/// compute alone. Their own subgroup widths (AMD's 64 in graphics stages) are not reproducible.
pub fn gpu_compat() -> Option<&'static str> {
    static C: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *C.get_or_init(|| match std::env::var("DS_GPU_COMPAT").ok().as_deref() {
        Some("amd") => Some("amd"),
        Some("intel") => Some("intel"),
        Some(v) => {
            crate::warn!("DS_GPU_COMPAT={v}: amd or intel; ignored");
            None
        }
        None => None,
    })
}

/// `DS_VRAM_CAP_MIB=<MiB>` (boot; simulation of a smaller GPU): the
/// device-local budget our decisions see (`Ctx::device_budget`: texture
/// eviction, and sizes derived from it) is at most this; allocations are
/// not limited by it.
/// `DS_SIM_VRAM_MIB=<n>` (a test of a card with less video memory, on this PC): this process's
/// room in the largest device-local heap ends at n MiB, as on a smaller card with other programs'
/// memory taken out: its allocations there fail past it (`Ctx::sim_alloc_fails`) and the budget
/// the port sees is n MiB (`device_budget`). Until 2026-11-21 night tests capped only that budget
/// (`DS_VRAM_CAP_MIB`): on a 24 GB card no allocation ever failed, so the failures a 12 GB card
/// met in the preparation (allocations refused, a lock held while one blocked) could not happen
/// here.
pub fn sim_vram_limit() -> Option<u64> {
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("DS_SIM_VRAM_MIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&m| m >= 512).map(|m| m << 20))
}

/// The card's video memory as the choices made by its size see it (image slabs, the mirror's
/// backing granule): the largest device-local heap (at most `vram_cap`), under `DS_SIM_VRAM_MIB`
/// the smaller card it stands for. Until 2026-11-22 the simulation kept this 24 GB card's slabs
/// and its 1 MiB backing granules, which a card under 16 GB never uses (~30% more mirror).
pub fn card_size() -> u64 {
    CTX.get().map_or(0, |c| c.device_heap_size()).min(sim_vram_limit().unwrap_or(u64::MAX))
}

/// `DS_SIM_HOST_MIB=<n>` (a test of a PC with less memory, 2026-11-22): the budget of the heap
/// that imported guest memory and the system-memory fallback use is at most n MiB, as on a PC
/// whose graphics share of memory (Windows gives about half the installed memory) is that small:
/// an import or a fallback allocation that would pass it fails (`sim_host_fails`), as the
/// driver's did on this PC when that heap reached its budget. A 16 GB PC: about 7600.
pub fn sim_host_limit() -> Option<u64> {
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("DS_SIM_HOST_MIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&m| m >= 1024).map(|m| m << 20))
}

/// Imports and fallback allocations failed by `DS_SIM_HOST_MIB` (with `DS_SIM_HOST_COUNT=1`: that
/// would have failed), and the most bytes one would have passed its budget by.
pub static SIM_HOST_FAILS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SIM_HOST_OVER_MAX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `DS_SIM_HOST_COUNT=1` with `DS_SIM_HOST_MIB` (2026-11-22): what would pass the simulated budget
/// is counted, not failed (`sim_host_fails`): a smaller PC's shortfall measured without failing
/// imports, which hung and reset the GPU here (GPU writes into ranges without imports never reach
/// guest memory, labels among them).
pub fn sim_host_count_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SIM_HOST_COUNT").ok().as_deref() == Some("1"))
}

/// Device-local allocations failed by `DS_SIM_VRAM_MIB`.
pub static SIM_VRAM_FAILS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The error an allocation failed by `DS_SIM_VRAM_MIB` returns (the driver's own words for it).
pub const SIM_OOM: vk::Result = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;

/// The refusal window after a video-memory allocation failed: for `GATE_MS` requests of at least
/// the smallest size that failed in it are refused with no driver call (`Ctx::refuse_alloc`),
/// and every path that allocates video memory notes its failures (`Ctx::note_alloc_fail`).
/// Without it a card out of room met the same failing allocation from every command and draw:
/// 12.8 million refusals in one 280 s session of a test with 7 GB of room (2026-11-21 night),
/// each a driver call, where a real failing vkAllocateMemory takes milliseconds while the system
/// pages; that session ran at 4 frames a second until its device was lost.
const GATE_MS: u64 = 250;
/// When the window began (`gate_clock_ms` + 1; 0: never) and the smallest size failed in it.
static GATE_AT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GATE_SIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
/// [requests the window refused, failures noted (the driver's or `DS_SIM_VRAM_MIB`'s)].
pub static GATE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn gate_clock_ms() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

/// Whether the refusal window refuses a video-memory request of `size` bytes now (counted).
pub fn gate_refuses(size: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let at = GATE_AT_MS.load(Relaxed);
    if at == 0 || size < GATE_SIZE.load(Relaxed) || gate_clock_ms() >= at + GATE_MS {
        return false;
    }
    GATE_STATS[0].fetch_add(1, Relaxed);
    true
}

/// The status line of failed and refused video-memory requests ("" with none).
pub fn gate_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let (refused, failed) = (GATE_STATS[0].load(Relaxed), GATE_STATS[1].load(Relaxed));
    let host = SIM_HOST_FAILS.load(Relaxed);
    if failed == 0 && host == 0 {
        return String::new();
    }
    let host = match host {
        0 => String::new(),
        n if sim_host_count_only() => format!("; imports and fallback allocations that would have failed under DS_SIM_HOST_MIB {n} (counted only; by at most {} MiB)", SIM_HOST_OVER_MAX.load(Relaxed) >> 20),
        n => format!("; imports and fallback allocations failed by DS_SIM_HOST_MIB {n} (by at most {} MiB)", SIM_HOST_OVER_MAX.load(Relaxed) >> 20),
    };
    format!("    video-memory allocations failed {failed} times ({} of them by DS_SIM_VRAM_MIB); {refused} requests refused with no driver call in the {GATE_MS} ms after a failure{host}\n", SIM_VRAM_FAILS.load(Relaxed))
}

pub fn vram_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| std::env::var("DS_VRAM_CAP_MIB").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&m| m >= 1024).map_or(u64::MAX, |m| m << 20))
}

/// A share of the device-local budget seen at the first call (`1/div` of
/// it, at most `max`; `max` without VK_EXT_memory_budget): fixed pools
/// (the shared copy cache, video-memory homes) sized for the GPU at hand.
pub fn budget_share(div: u64, max: u64) -> u64 {
    static BUDGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let b = *BUDGET.get_or_init(|| ctx().device_budget().map(|(_, b)| b));
    b.map_or(max, |b| (b / div).min(max))
}

/// `Ctx::heap_report` of the initialized context ("" before).
pub fn ctx_heaps() -> String {
    CTX.get().map(|c| c.heap_report()).unwrap_or_default()
}

/// Each memory type: its heap and property flags.
pub fn ctx_types() -> String {
    let Some(c) = CTX.get() else { return String::new() };
    let p = &c.mem_props;
    (0..p.memory_type_count as usize).map(|i| format!("{i}: heap {} {:?}", p.memory_types[i].heap_index, p.memory_types[i].property_flags)).collect::<Vec<_>>().join("; ")
}

/// The system-memory fallback (2026-11-22): when video memory is full, an image or a block of the
/// mirror's pages goes to a memory type outside the device-local heaps, system memory the GPU
/// reaches across the bus (slower to draw from, the picture right), instead of being left out or
/// bound as zeros. The 8-level tour in a 7 GB room had bound 55.5 million mirror placements as
/// zeros and failed 1.04 million image creations (the shadow atlas's depth targets among them),
/// its pictures wrong for good from the first failure. NVIDIA's sparse buffers take video memory
/// only (the mirror's type bits 0x2), so the mirror cannot use it: textures go to system memory
/// first while video memory is under pressure or over its mark (`images::create_texture`),
/// leaving video memory to the mirror, and come back once it has room (`images::maybe_evict`).
/// [images' bytes there now, images placed there in all, refusals for want of room or a failed
/// allocation, textures moved back to video memory].
pub static SYSTEM_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// The most images held there at once, bytes (the preparation's health line).
pub static SYSTEM_PEAK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The fallback's allocations live now (one per image, `slab::system`). At most a quarter of the
/// driver's limit on allocations (`system_count_limit`): AMD's Windows driver allows 4,096 in all,
/// a boot's images, slabs and buffers among them (the pre-release review of 2026-11-22).
pub static SYSTEM_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn system_count_limit() -> u64 {
    CTX.get().map_or(1024, |c| (c.limits.max_memory_allocation_count as u64 / 4).max(64))
}

/// Whether the fallback could take a texture of `size` bytes now: on, a type for images (or not
/// known yet), its room and its count not spent, not refusing that size. Textures in video memory
/// are evicted sooner under pressure only then (`images::Images::evict`): otherwise they would
/// come back in video memory, an upload for nothing (the pre-release review of 2026-11-22).
pub fn system_can_take(size: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    system_fallback()
        && !images_lack_system_type()
        && SYSTEM_STATS[0].load(Relaxed) + size + (256 << 20) <= SYSTEM_CAP
        && SYSTEM_COUNT.load(Relaxed) < system_count_limit()
        && !system_refusing(size)
}
/// The fallback's room: at most this much in all, and its heap's budget left with the larger of
/// `SYSTEM_SPARE` and an eighth of it free (`Ctx::system_alloc`): the title's imported memory is
/// in that heap too and grows at arrivals. 4 GiB and three quarters of the budget until the
/// 4-6 GB cards of 2026-11-22: in a 4 GB room textures and render targets need more there, and
/// the imports alone held 9.3 of this PC's 15.5 GB budget, which left the old rule ~2 GB.
const SYSTEM_CAP: u64 = 8 << 30;
const SYSTEM_SPARE: u64 = 2 << 30;
/// When the fallback last refused for want of room or a failed allocation (`gate_clock_ms`; 0:
/// never) and the smallest size refused since: for `GATE_MS` after it requests of at least that
/// size are refused with no driver call (`system_refusing`), smaller ones still tried.
static SYSTEM_REFUSED_AT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SYSTEM_REFUSED_SIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
/// Whether images have a memory type for the fallback (`note_image_bits`): 0 not known yet, 1 yes,
/// 2 none seen (a card whose heaps are all device-local: the refusal window's pre-check for images
/// stays as before the fallback, `images::create_inner`).
static IMAGE_SYSTEM_TYPE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `DS_SYSTEM_FALLBACK=0`: no system-memory fallback (as before 2026-11-22). On a card with room
/// the driver keeps the fallback's type-0 allocations in its free video memory (the 7 GB tours
/// of 2026-11-22 on the 24 GB card: heap 0's usage held them), so this card cannot show what
/// reading them across the bus costs; NVIDIA's optimal images take types 0 and 1 only (type bits
/// 0x3), none the driver must keep in system memory.
pub fn system_fallback() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SYSTEM_FALLBACK").ok().as_deref() != Some("0"))
}

/// Under `DS_SIM_VRAM_MIB`, the bytes this process placed in system memory for want of video
/// memory (`SYSTEM_STATS`), which the driver of a card with room may keep in its video memory
/// and count in the device-local heap's usage: on the smaller card the test stands for, they are
/// not there. 0 without the simulation (the driver's count is the truth then).
fn sim_system_bytes() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    if sim_vram_limit().is_none() {
        return 0;
    }
    SYSTEM_STATS[0].load(Relaxed)
}

/// Whether the fallback refuses a request of `size` bytes now: it is off, or it refused one of
/// that size or less in the last `GATE_MS` (`SYSTEM_REFUSED_SIZE`).
pub fn system_refusing(size: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let at = SYSTEM_REFUSED_AT.load(Relaxed);
    !system_fallback() || (at != 0 && size >= SYSTEM_REFUSED_SIZE.load(Relaxed) && gate_clock_ms() < at + GATE_MS)
}

fn system_refused(size: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    SYSTEM_STATS[2].fetch_add(1, Relaxed);
    let (now, at) = (gate_clock_ms(), SYSTEM_REFUSED_AT.load(Relaxed));
    if at == 0 || now >= at + GATE_MS {
        SYSTEM_REFUSED_SIZE.store(size, Relaxed);
        SYSTEM_REFUSED_AT.store(now, Relaxed);
    } else {
        SYSTEM_REFUSED_SIZE.fetch_min(size, Relaxed);
    }
}

/// Whether images are known to have no memory type for the fallback (`IMAGE_SYSTEM_TYPE`).
pub fn images_lack_system_type() -> bool {
    IMAGE_SYSTEM_TYPE.load(std::sync::atomic::Ordering::Relaxed) == 2
}

/// The fallback's status line ("" when it was never needed).
pub fn system_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let s: Vec<u64> = SYSTEM_STATS.iter().map(|c| c.load(Relaxed)).collect();
    if s[1] == 0 && s[2] == 0 {
        return String::new();
    }
    format!(
        "    system memory for want of video memory: images {} MiB now ({} placed there in all, at most {} MiB at once, {} textures moved back); refused {} times for want of room or a failed allocation (each refusing that size and more for {GATE_MS} ms)\n",
        s[0] >> 20,
        s[1],
        SYSTEM_PEAK.load(Relaxed) >> 20,
        s[3],
        s[2]
    )
}

/// Checkpoint marker kinds (high byte of the marker value).
pub mod mark {
    pub const DISPATCH: u64 = 1;
    pub const DRAW: u64 = 2;
    pub const SHADOW_IN: u64 = 3;
    pub const SHADOW_OUT: u64 = 4;
    pub const UPLOAD: u64 = 5;
    pub const BATCH: u64 = 6;
    /// A GPU-resolved draw run's multi-draw, by its number (`exec::draw`'s run log).
    pub const RUN: u64 = 7;
}

/// The device offers partitioned subgroup operations
/// (VK_NV_shader_subgroup_partitioned, enabled when offered): write marks are
/// then ORed per map word (`rdna::translate::Spec::marks_part`; the 19
/// hottest compute programs' code -12.8%, GPU busy -6.2% at the vista and
/// -2.8% at Doran, frames +1.1-1.2%: FINDINGS.md 2026-11-06).
pub fn subgroup_partitioned() -> bool {
    try_ctx().is_some_and(|c| c.partitioned)
}

/// Command labels are recorded (DS_GPU_LABELS=1, `Exec::label`).
pub fn labels_on() -> bool {
    try_ctx().is_some_and(|c| c.labels.is_some())
}

/// Begins a named label range in `cmd` (DS_GPU_LABELS=1).
pub fn label_begin(cmd: vk::CommandBuffer, name: &str) {
    if let Some(l) = try_ctx().and_then(|c| c.labels.as_ref()) {
        if defer::label_begin(cmd, name) {
            return;
        }
        defer::direct(cmd);
        let s = std::ffi::CString::new(name).unwrap_or_default();
        unsafe { l.cmd_begin_debug_utils_label(cmd, &vk::DebugUtilsLabelEXT::default().label_name(&s)) };
    }
}

/// Ends the innermost label range begun in `cmd`.
pub fn label_end(cmd: vk::CommandBuffer) {
    if let Some(l) = try_ctx().and_then(|c| c.labels.as_ref()) {
        if defer::label_end(cmd) {
            return;
        }
        defer::direct(cmd);
        unsafe { l.cmd_end_debug_utils_label(cmd) };
    }
}

/// Records a diagnostic checkpoint (DS_GPU_CHECKPOINTS=1): `kind` and a
/// 56-bit value (shader or guest address, sequence number).
/// Diagnostic checkpoints are recorded (VK_NV_device_diagnostic_checkpoints).
pub fn checkpoints_on() -> bool {
    try_ctx().is_some_and(|c| c.checkpoints.is_some())
}

pub fn checkpoint(cmd: vk::CommandBuffer, kind: u64, value: u64) {
    if let Some(c) = try_ctx() {
        if let Some(cp) = &c.checkpoints {
            let marker = (kind << 56) | (value & ((1 << 56) - 1));
            if defer::checkpoint(cmd, marker) {
                return;
            }
            defer::direct(cmd);
            unsafe { cp.cmd_set_checkpoint(cmd, marker as *const std::ffi::c_void) };
        }
    }
}

/// GPU address ranges of guest imports and video-memory homes, for device
/// fault reports: live ones, dropped ones (no new work uses them) and the
/// last freed ones.
struct GpuRange {
    buf: u64,
    va: u64,
    size: u64,
    what: String,
    dropped_ms: Option<f64>,
    freed_ms: Option<f64>,
}

static GPU_RANGES: Mutex<Vec<GpuRange>> = Mutex::new(Vec::new());

/// Remembers buffer `buf`'s GPU address range (`va` 0: none) as `what`.
pub fn note_gpu_range(buf: vk::Buffer, va: u64, size: u64, what: impl FnOnce() -> String) {
    use ash::vk::Handle;
    if va == 0 {
        return;
    }
    GPU_RANGES.lock().push(GpuRange { buf: buf.as_raw(), va, size, what: what(), dropped_ms: None, freed_ms: None });
}

/// Buffer `buf` is no longer used for new work (freed once its batches end).
pub fn note_gpu_dropped(buf: vk::Buffer) {
    use ash::vk::Handle;
    let t = crate::log::elapsed_ms();
    if let Some(r) = GPU_RANGES.lock().iter_mut().rev().find(|r| r.buf == buf.as_raw() && r.freed_ms.is_none()) {
        r.dropped_ms.get_or_insert(t);
    }
}

/// Buffer `buf` was destroyed (its range is kept among the last freed).
pub fn note_gpu_freed(buf: vk::Buffer) {
    use ash::vk::Handle;
    let t = crate::log::elapsed_ms();
    let mut g = GPU_RANGES.lock();
    if let Some(r) = g.iter_mut().rev().find(|r| r.buf == buf.as_raw() && r.freed_ms.is_none()) {
        r.freed_ms = Some(t);
    }
    let freed = g.iter().filter(|r| r.freed_ms.is_some()).count();
    if freed > 4096 {
        let mut drop_n = freed - 4096;
        g.retain(|r| {
            if drop_n > 0 && r.freed_ms.is_some() {
                drop_n -= 1;
                false
            } else {
                true
            }
        });
    }
}

/// Where a GPU address lies among the noted ranges.
fn describe_address(a: u64) -> String {
    let g = GPU_RANGES.lock();
    let now = crate::log::elapsed_ms();
    let mut hits: Vec<String> = g
        .iter()
        .filter(|r| r.va <= a && a < r.va + r.size)
        .map(|r| {
            let state = match (r.dropped_ms, r.freed_ms) {
                (_, Some(f)) => format!("FREED {:.0} ms before the report", now - f),
                (Some(d), None) => format!("dropped {:.0} ms before the report, not freed", now - d),
                (None, None) => "live".into(),
            };
            format!("{} +{:#x} ({state})", r.what, a - r.va)
        })
        .collect();
    // Front end 2's mirror (a sparse buffer over the guest span; GPU-resolved reads go there at
    // guest offsets): the guest address it stands for.
    let m = crate::gpu::resident::mirror_address();
    if m != 0 && m <= a && a < m + (rdna::translate::PT_END - rdna::translate::PT_BASE) {
        hits.push(format!("in front end 2's mirror at guest {:#x}", rdna::translate::PT_BASE + (a - m)));
    }
    if hits.is_empty() {
        // The nearest noted ranges below and above.
        let below = g.iter().filter(|r| r.va + r.size <= a).max_by_key(|r| r.va + r.size).map(|r| format!("{:#x} past the end of {}", a - (r.va + r.size), r.what));
        let above = g.iter().filter(|r| r.va > a).min_by_key(|r| r.va).map(|r| format!("{:#x} before {}", r.va - a, r.what));
        hits.push(format!("in no noted range ({} ranges noted; {})", g.len(), [below, above].into_iter().flatten().collect::<Vec<_>>().join("; ")));
    }
    hits.join(" | ")
}

/// The device's account of a loss (VK_EXT_device_fault): description,
/// faulting addresses with the noted ranges they fall in, vendor codes; the
/// vendor binary goes to `gpu-fault.bin` in the run directory. Computed once.
pub fn fault_report() -> String {
    static REPORT: OnceLock<String> = OnceLock::new();
    REPORT
        .get_or_init(|| {
            let Some(c) = try_ctx() else { return String::new() };
            let Some(f) = &c.fault else { return "(no VK_EXT_device_fault)".into() };
            unsafe {
                let mut counts = vk::DeviceFaultCountsEXT::default();
                let r = (f.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, std::ptr::null_mut());
                if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                    return format!("vkGetDeviceFaultInfoEXT: {r}");
                }
                let mut addrs = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
                let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
                let mut bin = vec![0u8; counts.vendor_binary_size as usize];
                let mut info = vk::DeviceFaultInfoEXT::default();
                info.p_address_infos = if addrs.is_empty() { std::ptr::null_mut() } else { addrs.as_mut_ptr() };
                info.p_vendor_infos = if vendors.is_empty() { std::ptr::null_mut() } else { vendors.as_mut_ptr() };
                info.p_vendor_binary_data = if bin.is_empty() { std::ptr::null_mut() } else { bin.as_mut_ptr().cast() };
                let r = (f.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, &mut info);
                if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                    return format!("vkGetDeviceFaultInfoEXT: {r}");
                }
                let desc = CStr::from_ptr(info.description.as_ptr()).to_string_lossy().into_owned();
                let mut out = vec![format!("device fault: '{desc}'")];
                for a in addrs.iter().take(counts.address_info_count as usize) {
                    out.push(format!("{:?} at {:#x} (precision {:#x}): {}", a.address_type, a.reported_address, a.address_precision, describe_address(a.reported_address)));
                }
                for v in vendors.iter().take(counts.vendor_info_count as usize) {
                    let d = CStr::from_ptr(v.description.as_ptr()).to_string_lossy().into_owned();
                    out.push(format!("vendor '{d}' code {:#x} data {:#x}", v.vendor_fault_code, v.vendor_fault_data));
                }
                // (The vendor's binary in development checkouts only: a player's log folder is
                // what they attach to a public report.)
                if !bin.is_empty() && crate::paths().layout == dsconfig::paths::Layout::Dev {
                    if let Some(dir) = crate::run_dir() {
                        let p = dir.join("gpu-fault.bin");
                        if std::fs::write(&p, &bin).is_ok() {
                            out.push(format!("vendor binary ({} bytes): {}", bin.len(), p.display()));
                        }
                    }
                }
                out.join("; ")
            }
        })
        .clone()
}

/// The queue's last checkpoints per pipeline stage, after a device loss
/// (and the device's fault report).
pub fn checkpoint_report() -> String {
    let fault = fault_report();
    let Some(c) = try_ctx() else { return String::new() };
    let Some(cp) = &c.checkpoints else { return format!("(DS_GPU_CHECKPOINTS=1 names the failing command); {fault}") };
    let q = c.queue.lock();
    unsafe {
        let n = cp.get_queue_checkpoint_data_len(*q);
        let mut v = vec![vk::CheckpointDataNV::default(); n];
        cp.get_queue_checkpoint_data(*q, &mut v);
        v.iter()
            .map(|d| {
                let m = d.p_checkpoint_marker as u64;
                let kind = match m >> 56 {
                    mark::DISPATCH => "dispatch cs",
                    mark::DRAW => "draw ps",
                    mark::SHADOW_IN => "shadow copy in",
                    mark::SHADOW_OUT => "shadow copy out",
                    mark::UPLOAD => "image upload",
                    mark::BATCH => "batch",
                    mark::RUN => "GPU-resolved run #",
                    _ => "?",
                };
                format!("{:?}: {kind} {:#x}", d.stage, m & ((1 << 56) - 1))
            })
            .chain(std::iter::once(fault))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl Ctx {
    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want))
    }

    /// A memory type for `bits` in a heap that is not device-local (`SYSTEM_STATS`): one the CPU
    /// cannot map first (NVIDIA's type 0), else any. None when the fallback is off or every heap
    /// is device-local (integrated GPUs, whose memory is system memory already).
    pub fn system_type(&self, bits: u32) -> Option<u32> {
        if !system_fallback() {
            return None;
        }
        let p = &self.mem_props;
        let ok = |i: u32| bits & (1 << i) != 0 && !p.memory_heaps[p.memory_types[i as usize].heap_index as usize].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        (0..p.memory_type_count)
            .find(|&i| ok(i) && !p.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
            .or_else(|| (0..p.memory_type_count).find(|&i| ok(i)))
    }

    /// `DS_SIM_HOST_MIB`: whether `size` more bytes in memory type `ty` would pass the simulated
    /// budget of its heap (a heap that is not device-local only), counted in `SIM_HOST_FAILS`.
    pub fn sim_host_fails(&self, ty: u32, size: u64) -> bool {
        let Some(limit) = sim_host_limit() else { return false };
        if !self.memory_budget || ty >= self.mem_props.memory_type_count {
            return false;
        }
        let heap = self.mem_props.memory_types[ty as usize].heap_index as usize;
        if self.mem_props.memory_heaps[heap].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            return false;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        drop(p2);
        let need = budget.heap_usage[heap].saturating_add(size);
        let fails = need > limit;
        if fails {
            let n = SIM_HOST_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            SIM_HOST_OVER_MAX.fetch_max(need - limit, std::sync::atomic::Ordering::Relaxed);
            if n < 8 || n % 1000 == 0 {
                crate::info!(
                    "gpu: DS_SIM_HOST_MIB: {} MiB more would pass the simulated budget ({} MiB used of {}){} ({} so far)",
                    size >> 20,
                    budget.heap_usage[heap] >> 20,
                    limit >> 20,
                    if sim_host_count_only() { ", counted only" } else { "" },
                    n + 1
                );
            }
        }
        fails && !sim_host_count_only()
    }

    /// An image's memory type bits seen (`images::create_inner`): whether images have a type for
    /// the fallback (`IMAGE_SYSTEM_TYPE`; one that has wins over one that has not).
    pub fn note_image_bits(&self, bits: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        if IMAGE_SYSTEM_TYPE.load(Relaxed) == 1 || !system_fallback() {
            return;
        }
        IMAGE_SYSTEM_TYPE.store(if self.system_type(bits).is_some() { 1 } else { 2 }, Relaxed);
    }

    /// Memory of `size` bytes from the system-memory fallback for requirements' type bits `bits`
    /// (device addresses with `addressed`): None when it has no type for them, or no room
    /// (`SYSTEM_CAP`, its heap's spare room `SYSTEM_SPARE`) or the allocation failed, these two
    /// counted and refusing that size and more for `GATE_MS` (`system_refusing`, no driver call
    /// then). A request with no type refuses nothing else (in the fallback's first form the
    /// mirror, which has none on NVIDIA, kept images from it so for 250 ms at each failure). The
    /// caller counts what it placed (`SYSTEM_STATS`).
    pub fn system_alloc(&self, bits: u32, size: u64, addressed: bool) -> Option<(vk::DeviceMemory, u32)> {
        use std::sync::atomic::Ordering::Relaxed;
        let ty = self.system_type(bits)?;
        if system_refusing(size) {
            return None;
        }
        if SYSTEM_COUNT.load(Relaxed) >= system_count_limit() {
            SYSTEM_STATS[2].fetch_add(1, Relaxed);
            return None;
        }
        let held = SYSTEM_STATS[0].load(Relaxed);
        let mut room = held + size <= SYSTEM_CAP;
        if room && self.memory_budget {
            let heap = self.mem_props.memory_types[ty as usize].heap_index as usize;
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
            drop(p2);
            let heap_budget = budget.heap_budget[heap].min(sim_host_limit().unwrap_or(u64::MAX));
            let spare = SYSTEM_SPARE.max(heap_budget / 8);
            room = budget.heap_usage[heap] + size + spare <= heap_budget;
        }
        if !room {
            system_refused(size);
            return None;
        }
        let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty);
        let ai = if addressed && self.device_addresses { ai.push_next(&mut fl) } else { ai };
        let t0 = std::time::Instant::now();
        let r = unsafe { self.device.allocate_memory(&ai, None) };
        note_alloc("system memory for want of video memory", size, t0);
        match r {
            Ok(m) => Some((m, ty)),
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::warn!("gpu: system memory for want of video memory failed ({size} bytes, memory type {ty}): {e}"));
                system_refused(size);
                None
            }
        }
    }

    pub fn alloc(&self, req: vk::MemoryRequirements, want: vk::MemoryPropertyFlags) -> Result<vk::DeviceMemory, String> {
        let ty = self.memory_type(req.memory_type_bits, want).ok_or("no suitable memory type")?;
        if self.refuse_alloc(ty, req.size) {
            return Err(format!("vkAllocateMemory({}): {SIM_OOM} (refused: video memory ran out in the last {GATE_MS} ms, or DS_SIM_VRAM_MIB)", req.size));
        }
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        let t0 = std::time::Instant::now();
        let r = unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| {
            self.note_alloc_fail(ty, req.size);
            format!("vkAllocateMemory({}): {e}", req.size)
        });
        note_alloc("memory", req.size, t0);
        r
    }

    /// `alloc` for buffer memory: with bufferDeviceAddress the buffers bound
    /// to it may be addressed by shaders (`buffer_address`).
    pub fn alloc_buffer_mem(&self, req: vk::MemoryRequirements, want: vk::MemoryPropertyFlags) -> Result<vk::DeviceMemory, String> {
        if !self.device_addresses {
            return self.alloc(req, want);
        }
        let ty = self.memory_type(req.memory_type_bits, want).ok_or("no suitable memory type")?;
        if self.refuse_alloc(ty, req.size) {
            return Err(format!("vkAllocateMemory({}): {SIM_OOM} (refused: video memory ran out in the last {GATE_MS} ms, or DS_SIM_VRAM_MIB)", req.size));
        }
        let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut fl);
        let t0 = std::time::Instant::now();
        let r = unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| {
            self.note_alloc_fail(ty, req.size);
            format!("vkAllocateMemory({}): {e}", req.size)
        });
        note_alloc("buffer memory", req.size, t0);
        r
    }

    /// Usage flags for a buffer that shaders may address (`Spec::addresses`).
    pub fn addressable(&self, usage: vk::BufferUsageFlags) -> vk::BufferUsageFlags {
        if self.device_addresses { usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS } else { usage }
    }

    /// The device address of `buf` (0 without bufferDeviceAddress). Every
    /// buffer our helpers create is addressable.
    #[inline]
    pub fn buffer_address(&self, buf: vk::Buffer) -> u64 {
        if !self.device_addresses {
            return 0;
        }
        unsafe { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) }
    }

    /// Device-local buffer with a GPU address (0 without bufferDeviceAddress):
    /// video-memory homes, named in device fault reports.
    pub fn device_buffer_addressed(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        if !self.device_addresses {
            let (b, m) = self.device_buffer(size, usage)?;
            return Ok((b, m, 0));
        }
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let Some(ty) = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
                self.device.destroy_buffer(buf, None);
                return Err("no suitable memory type".into());
            };
            if self.refuse_alloc(ty, req.size) {
                self.device.destroy_buffer(buf, None);
                return Err(format!("vkAllocateMemory({}): {SIM_OOM} (refused: video memory ran out in the last {GATE_MS} ms, or DS_SIM_VRAM_MIB)", req.size));
            }
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut fl);
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.note_alloc_fail(ty, req.size);
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory({}): {e}", req.size));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind buffer: {e}"));
            }
            let va = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            Ok((buf, mem, va))
        }
    }

    /// `bi` with the buffer shared by the families of the main, copy and
    /// compute queues (concurrent; exclusive with the main queue's alone).
    fn shared<'a>(&'a self, bi: vk::BufferCreateInfo<'a>) -> vk::BufferCreateInfo<'a> {
        if self.families.len() > 1 {
            bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&self.families)
        } else {
            bi.sharing_mode(vk::SharingMode::EXCLUSIVE)
        }
    }

    /// Device-local buffer (not host-visible).
    pub fn device_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // (The buffer destroyed when its memory fails: a refused allocation retried by every
            // batch had left one buffer behind each time.)
            let mem = match self.alloc_buffer_mem(req, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(e);
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind buffer: {e}"));
            }
            Ok((buf, mem))
        }
    }

    /// Imports a dma-buf made from a memfd by udmabuf, when RADV refuses the
    /// host pointer. The fd is not owned here: the caller closes it.
    #[cfg(target_os = "linux")]
    pub fn import_dmabuf(&self, fd: i32, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        let usage = self.addressable(usage);
        let ht = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        unsafe {
            let mut ebi = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(usage)).push_next(&mut ebi);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer(dma-buf): {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let Some(ty) = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty()))
            else {
                self.device.destroy_buffer(buf, None);
                return Err("no memory type for a dma-buf import".into());
            };
            let mut ifd = vk::ImportMemoryFdInfoKHR::default().handle_type(ht).fd(fd);
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut ifd);
            let ai = if self.device_addresses { ai.push_next(&mut fl) } else { ai };
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory(dma-buf {size:#x}): {e}"));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind dma-buf: {e}"));
            }
            let va = if self.device_addresses {
                self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf))
            } else { 0 };
            Ok((buf, mem, va))
        }
    }

    /// Imports existing host memory [ptr, ptr+size) (page aligned) as a GPU
    /// buffer (VK_EXT_external_memory_host): the GPU reads and writes it in place.
    /// The third value is the buffer's GPU address (0 without bufferDeviceAddress).
    fn import_host_ptr(&self, ptr: u64, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        let usage = if self.device_addresses { usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS } else { usage };
        let ext = self.ext_host.as_ref().ok_or("VK_EXT_external_memory_host not available")?;
        let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            let r = (ext.fp().get_memory_host_pointer_properties_ext)(self.device.handle(), ht, ptr as *const std::ffi::c_void, &mut props);
            if r != vk::Result::SUCCESS {
                return Err(format!("vkGetMemoryHostPointerPropertiesEXT: {r}"));
            }
            let mut ebi = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(usage)).push_next(&mut ebi);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let bits = req.memory_type_bits & props.memory_type_bits;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = self
                .memory_type(bits, hv | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.memory_type(bits, hv))
                .or_else(|| self.memory_type(bits, vk::MemoryPropertyFlags::empty()));
            let Some(ty) = ty else {
                self.device.destroy_buffer(buf, None);
                return Err(format!("no memory type for imported host memory (bits {bits:#x})"));
            };
            if self.sim_host_fails(ty, size) {
                self.device.destroy_buffer(buf, None);
                return Err(format!("vkAllocateMemory(import {size:#x}): {} (DS_SIM_HOST_MIB)", vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
            }
            let mut imp = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(ht).host_pointer(ptr as *mut std::ffi::c_void);
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty).push_next(&mut imp);
            let ai = if self.device_addresses { ai.push_next(&mut fl) } else { ai };
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory(import {size:#x}): {e}"));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind imported buffer: {e}"));
            }
            let va = if self.device_addresses { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) } else { 0 };
            Ok((buf, mem, va))
        }
    }

    /// Host pointer first; on RADV's refusal, try the dma-buf route before
    /// giving up. Both paths and every rejection are counted for the status line.
    pub fn import_host(&self, ptr: u64, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        match self.import_host_ptr(ptr, size, usage) {
            Ok(r) => {
                #[cfg(target_os = "linux")]
                crate::platform::os::note_host(true);
                Ok(r)
            }
            Err(e) => {
                #[cfg(target_os = "linux")]
                crate::platform::os::note_host(false);
                #[cfg(target_os = "linux")]
                if self.dmabuf {
                    if let Some((fd, n)) = crate::platform::os::host_import_source(ptr, size) {
                        match self.import_dmabuf(fd, n, usage) {
                            Ok(r) => {
                                crate::platform::os::note_dma(true);
                                unsafe { libc::close(fd) };
                                return Ok(r);
                            }
                            Err(err) => {
                                crate::platform::os::note_dma(false);
                                crate::warn!("dma-buf import of {ptr:#x}+{size:#x} failed: {err}");
                            }
                        }
                        unsafe { libc::close(fd) };
                    }
                }
                #[cfg(not(target_os = "linux"))]
                let _ = self.dmabuf;
                Err(e)
            }
        }
    }

    /// Device-local buffer the CPU writes through its mapping (the PCIe BAR
    /// window; writes are write-combined, reads by the CPU are very slow).
    /// GPU reads stay in video memory, unlike `host_buffer`.
    pub fn bar_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = match self.alloc_buffer_mem(req, want) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(e);
                }
            };
            self.bind_map(buf, mem, size)
        }
    }

    /// `mem` bound to `buf` and mapped, both freed when either fails (`bar_buffer`, `host_buffer`,
    /// `upload_buffer`: such a failure had left them behind; on cards whose system memory is
    /// their video memory the refusal window can refuse these too, at a storm's rate; the review
    /// of 2026-11-21 night).
    unsafe fn bind_map(&self, buf: vk::Buffer, mem: vk::DeviceMemory, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        let r = unsafe { self.device.bind_buffer_memory(buf, mem, 0) }.map_err(|e| format!("bind buffer: {e}")).and_then(|_| unsafe { self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) }.map_err(|e| format!("map: {e}")));
        match r {
            Ok(p) => Ok((buf, mem, p as *mut u8)),
            Err(e) => {
                unsafe {
                    self.device.destroy_buffer(buf, None);
                    self.device.free_memory(mem, None);
                }
                Err(e)
            }
        }
    }

    /// Host-visible, coherent buffer (staging / per-draw data).
    pub fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // Prefer cached host memory: results are read back by the CPU
            // (buffer write-back), which is very slow from write-combined memory.
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = match self.alloc_buffer_mem(req, hv | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc_buffer_mem(req, hv)) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(e);
                }
            };
            self.bind_map(buf, mem, size)
        }
    }

    /// Buffer in system memory (never the PCIe BAR window's video memory)
    /// that the CPU fills and the GPU reads across PCIe: write-combined
    /// where the driver offers it, else cached.
    pub fn upload_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let pick = |not: vk::MemoryPropertyFlags| {
                (0..self.mem_props.memory_type_count).find(|&i| {
                    let f = self.mem_props.memory_types[i as usize].property_flags;
                    req.memory_type_bits & (1 << i) != 0 && f.contains(hv) && !f.intersects(not)
                })
            };
            let Some(ty) = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)) else {
                self.device.destroy_buffer(buf, None);
                return Err("no host-visible system memory type".into());
            };
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
            if self.device_addresses {
                ai = ai.push_next(&mut fl);
            }
            let t0 = std::time::Instant::now();
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory({}): {e}", req.size));
                }
            };
            note_alloc("upload memory", req.size, t0);
            self.bind_map(buf, mem, size)
        }
    }

    /// What this process may still allocate (its budget less its usage,
    /// VK_EXT_memory_budget) in the heap of the CPU-visible video memory
    /// `bar_buffer` takes (the PCIe BAR window: ~214 MiB on this machine,
    /// shared with the driver and every other process). None without the
    /// extension or without such memory.
    pub fn bar_room(&self) -> Option<u64> {
        if !self.memory_budget {
            return None;
        }
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ty = (0..self.mem_props.memory_type_count).find(|&i| self.mem_props.memory_types[i as usize].property_flags.contains(want))?;
        let heap = self.mem_props.memory_types[ty as usize].heap_index as usize;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        drop(p2);
        Some(budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap]))
    }

    /// A GPU timestamp and the performance counter read together (None
    /// without VK_EXT_calibrated_timestamps).
    pub fn calibrate(&self) -> Option<(u64, i64)> {
        let cal = self.calibrated.as_ref()?;
        let info = [vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE), vk::CalibratedTimestampInfoEXT::default().time_domain(window::time_domain())];
        let (t, _) = unsafe { cal.get_calibrated_timestamps(&info) }.ok()?;
        Some((t[0], t[1] as i64))
    }

    pub fn submit(&self, cmd: vk::CommandBuffer, wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        self.submit_cmds(&[cmd], wait, signal, fence)
    }

    /// Submits command buffers to the main queue, or the copy queue, with
    /// timeline semaphore waits (semaphore, value, stages that wait) and
    /// signals (semaphore, value).
    pub fn submit_timeline(&self, on_copy: bool, cmds: &[vk::CommandBuffer], waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)], signals: &[(vk::Semaphore, u64)], fence: vk::Fence) -> Result<(), String> {
        self.submit_timeline_on(on_copy, false, cmds, waits, signals, fence)
    }

    /// `submit_timeline`, on the compute family's queue when `on_compute`
    /// (ordered async compute).
    pub fn submit_timeline_on(&self, on_copy: bool, on_compute: bool, cmds: &[vk::CommandBuffer], waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)], signals: &[(vk::Semaphore, u64)], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<vk::CommandBufferSubmitInfo> = cmds.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect();
        let ws: Vec<vk::SemaphoreSubmitInfo> = waits.iter().map(|&(s, v, st)| vk::SemaphoreSubmitInfo::default().semaphore(s).value(v).stage_mask(st)).collect();
        let mut ss: Vec<vk::SemaphoreSubmitInfo> = signals.iter().map(|&(s, v)| vk::SemaphoreSubmitInfo::default().semaphore(s).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();
        if on_copy {
            let t0 = std::time::Instant::now();
            let q = self.copy.as_ref().ok_or("no copy queue")?.queue.lock();
            let t1 = std::time::Instant::now();
            let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
            let r = unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"));
            drop(q);
            note_submit("vkQueueSubmit2 (copy queue)", t0, t1);
            return r;
        }
        if on_compute {
            let t0 = std::time::Instant::now();
            let q = self.compute.as_ref().ok_or("no compute queue")?.queue.lock();
            let t1 = std::time::Instant::now();
            let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
            let r = unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2 (compute queue): {e}"));
            drop(q);
            note_submit("vkQueueSubmit2 (compute queue)", t0, t1);
            return r;
        }
        // (`DS_IMPORT_LOG=1`, diagnosis: the waits not yet signalled when
        // the submission is made, logged with a slow one: a wait before its
        // signal may hold the driver's submission up.)
        static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let log = *LOG.get_or_init(|| std::env::var("DS_IMPORT_LOG").ok().as_deref() == Some("1"));
        let unsignalled: Vec<String> = if log {
            waits
                .iter()
                .filter_map(|&(s, v, _)| {
                    let now = unsafe { self.device.get_semaphore_counter_value(s) }.unwrap_or(u64::MAX);
                    let name = if s == self.cpu_done { "cpu effects" } else if s == self.batches { "main batches" } else if self.compute.as_ref().is_some_and(|c| c.batches == s) { "compute queue" } else { "other" };
                    (now < v).then(|| format!("{name} {v} (at {now})"))
                })
                .collect()
        } else {
            Vec::new()
        };
        let t0 = std::time::Instant::now();
        let q = self.queue.lock();
        let t1 = std::time::Instant::now();
        ss.extend(self.progress_signal());
        let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
        let r = unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"));
        drop(q);
        note_submit("vkQueueSubmit2", t0, t1);
        if log && t0.elapsed().as_micros() >= 2000 {
            crate::info!("gpu: the slow submission had {} command buffers, {} waits ({} not signalled when made: {})", cmds.len(), waits.len(), unsignalled.len(), unsignalled.join(", "));
        }
        r
    }

    /// With the compute family's queue: the main queue's progress signal
    /// for a submission to it, made under its lock (values in submission
    /// order).
    fn progress_signal(&self) -> Option<vk::SemaphoreSubmitInfo<'static>> {
        self.compute.as_ref()?;
        let v = self.g_count.fetch_add(1, Ordering::AcqRel) + 1;
        Some(vk::SemaphoreSubmitInfo::default().semaphore(self.g_progress).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
    }

    /// The main queue's progress value of its latest submission (a batch
    /// on the compute family's queue waits for it).
    pub fn g_progress_value(&self) -> u64 {
        self.g_count.load(Ordering::Acquire)
    }

    /// Submits batches to the main queue in one call, in order: each its
    /// command buffers, an optional wait for the copy queue's timeline value,
    /// an optional wait for the compute family's queue's timeline value
    /// (ordered async compute: the compute batch it depends on), and a
    /// signal of `batches` with its sequence; `fence` on the call (the last
    /// batch's).
    pub fn submit_batches(&self, batches: &[(&[vk::CommandBuffer], u64, u64, u64)], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<Vec<vk::CommandBufferSubmitInfo>> = batches.iter().map(|b| b.0.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect()).collect();
        let copied = self.copy.as_ref().map(|q| q.copied);
        let cq = self.compute.as_ref().map(|q| q.batches);
        let ws: Vec<Vec<vk::SemaphoreSubmitInfo>> = batches
            .iter()
            .map(|b| {
                let mut w = Vec::new();
                if let (Some(s), true) = (copied, b.1 > 0) {
                    w.push(vk::SemaphoreSubmitInfo::default().semaphore(s).value(b.1).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
                }
                if let (Some(s), true) = (cq, b.3 > 0) {
                    w.push(vk::SemaphoreSubmitInfo::default().semaphore(s).value(b.3).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
                }
                w
            })
            .collect();
        let t0 = std::time::Instant::now();
        let q = self.queue.lock();
        let t1 = std::time::Instant::now();
        let ss: Vec<Vec<vk::SemaphoreSubmitInfo>> = (0..batches.len())
            .map(|k| {
                let mut v = vec![vk::SemaphoreSubmitInfo::default().semaphore(self.batches).value(batches[k].2).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                // (The progress signal on the last: its scope holds all
                // earlier submissions.)
                if k + 1 == batches.len() {
                    v.extend(self.progress_signal());
                }
                v
            })
            .collect();
        let infos: Vec<vk::SubmitInfo2> = (0..batches.len()).map(|k| vk::SubmitInfo2::default().command_buffer_infos(&cbs[k]).wait_semaphore_infos(&ws[k]).signal_semaphore_infos(&ss[k])).collect();
        let r = unsafe { self.device.queue_submit2(*q, &infos, fence) }.map_err(|e| format!("vkQueueSubmit2 ({} batches): {e}", batches.len()));
        drop(q);
        note_submit("vkQueueSubmit2 (batches)", t0, t1);
        r
    }

    /// Waits until the main queue's batch `seq` completed (`batches`);
    /// false on a timeout or error.
    /// The batch timeline's value now: the highest batch sequence the main
    /// queue has completed (every earlier one of its batches too).
    pub fn batch_value(&self) -> u64 {
        unsafe { self.device.get_semaphore_counter_value(self.batches) }.unwrap_or(0)
    }

    pub fn wait_batch(&self, seq: u64, timeout_ns: u64) -> bool {
        self.wait_batch_result(seq, timeout_ns).is_ok()
    }

    /// `wait_batch` telling a timeout (`vk::Result::TIMEOUT`) from an error.
    pub fn wait_batch_result(&self, seq: u64, timeout_ns: u64) -> Result<(), vk::Result> {
        let s = [self.batches];
        let v = [seq];
        unsafe { self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), timeout_ns) }
    }

    /// The compute family's queue's batch timeline now (the highest batch
    /// sequence it completed, every earlier one of its batches too).
    pub fn batch_value_c(&self) -> u64 {
        self.compute.as_ref().map_or(0, |q| unsafe { self.device.get_semaphore_counter_value(q.batches) }.unwrap_or(0))
    }

    /// `wait_batch` for a batch on the compute family's queue.
    pub fn wait_batch_c(&self, seq: u64, timeout_ns: u64) -> bool {
        self.wait_batch_c_result(seq, timeout_ns).is_ok()
    }

    /// `wait_batch_c` telling a timeout (`vk::Result::TIMEOUT`) from an error.
    pub fn wait_batch_c_result(&self, seq: u64, timeout_ns: u64) -> Result<(), vk::Result> {
        let Some(q) = self.compute.as_ref() else { return Err(vk::Result::ERROR_UNKNOWN) };
        let s = [q.batches];
        let v = [seq];
        unsafe { self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), timeout_ns) }
    }

    /// Submits command buffers that execute in order (binary semaphores).
    pub fn submit_cmds(&self, cmds: &[vk::CommandBuffer], wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<vk::CommandBufferSubmitInfo> = cmds.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect();
        let ws: Vec<vk::SemaphoreSubmitInfo> = wait.iter().map(|&(s, st)| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::from_raw(st.as_raw() as u64))).collect();
        let mut ss: Vec<vk::SemaphoreSubmitInfo> = signal.iter().map(|&s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();
        let q = self.queue.lock();
        ss.extend(self.progress_signal());
        let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
        unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"))
    }
}

#[cfg(test)]
mod validation_switch_tests {
    #[test]
    fn subgroup_policy_includes_blackwell_boundary() {
        assert!(super::modes_checked_on(0x10de, 0x2684));
        assert!(super::modes_checked_on(0x10de, 0x2b7f));
        assert_eq!(super::modes_checked_on(0x10de, 0x2b80), cfg!(target_os = "linux"));
        assert_eq!(super::modes_checked_on(0x10de, 0x2bb1), cfg!(target_os = "linux"));
        assert!(!super::modes_checked_on(0x1002, 0x744c));
        assert!(!super::modes_checked_on(0x8086, 0x56a0));
        // Linux NVIDIA subgroup support must not opt Blackwell out of float preservation.
        assert!(!super::nan_preserve_default(0x10de, 0x2684));
        assert!(super::nan_preserve_default(0x10de, 0x2b80));
        assert!(super::nan_preserve_default(0x10de, 0x2bb1));
        assert!(super::nan_preserve_default(0x1002, 0x744c));
        assert!(super::nan_preserve_default(0x8086, 0x56a0));
    }

    #[test]
    fn explicit_zero_disables_validation() {
        use std::ffi::OsStr;
        assert!(!super::validation_requested(None));
        assert!(!super::validation_requested(Some(OsStr::new("0"))));
        assert!(super::validation_requested(Some(OsStr::new("1"))));
        assert!(super::validation_requested(Some(OsStr::new(""))));
    }
}
