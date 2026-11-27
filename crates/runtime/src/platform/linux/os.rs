//! Linux implementations of the small host API surface used by the shared renderer.
//! Names retain the legacy call sites; Windows reexports the original windows-sys APIs.
//! Memory replacement is restricted to reservations owned by this module.
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
pub type HANDLE = *mut c_void;
pub const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;
pub const MEM_DECOMMIT: u32 = 0x4000;
pub const MEM_RELEASE: u32 = 0x8000;
pub const MEM_FREE: u32 = 0x10000;
pub const MEM_PRIVATE: u32 = 0x20000;
pub const MEM_MAPPED: u32 = 0x40000;
pub const MEM_IMAGE: u32 = 0x1000000;
pub const MEM_WRITE_WATCH: u32 = 0x200000;
pub const MEM_RESERVE_PLACEHOLDER: u32 = 0x40000;
pub const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
pub const MEM_PRESERVE_PLACEHOLDER: u32 = 2;
pub const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
pub const SEC_RESERVE: u32 = 0x4000000;
pub const PAGE_NOACCESS: u32 = 1;
pub const PAGE_READONLY: u32 = 2;
pub const PAGE_READWRITE: u32 = 4;
pub const PAGE_WRITECOPY: u32 = 8;
pub const PAGE_EXECUTE: u32 = 0x10;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;
pub const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;
pub const PAGE_GUARD: u32 = 0x100;
const PAGE: usize = 4096;
fn prot(p: u32) -> i32 {
    match p & 0xff {
        PAGE_READONLY => libc::PROT_READ,
        PAGE_READWRITE | PAGE_WRITECOPY => 3,
        PAGE_EXECUTE => libc::PROT_EXEC,
        PAGE_EXECUTE_READ => 5,
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => 7,
        _ => 0,
    }
}
fn writable(p: u32) -> bool {
    prot(p) & libc::PROT_WRITE != 0
}
#[derive(Clone)]
struct Region {
    end: usize,
    allocation: usize,
    protection: u32,
    committed: bool,
    mapped: bool,
    watch: bool,
    /// memfd and view offset, for the dma-buf import of a guest range.
    section: Option<(i32, u64)>,
}
struct Maps {
    regions: BTreeMap<usize, Region>,
    dirty: BTreeSet<usize>,
}
static MAPS: Mutex<Maps> = Mutex::new(Maps {
    regions: BTreeMap::new(),
    dirty: BTreeSet::new(),
});
impl Maps {
    fn find(&self, at: usize) -> Option<(usize, &Region)> {
        self.regions
            .range(..=at)
            .next_back()
            .filter(|(_, r)| at < r.end)
            .map(|(&a, r)| (a, r))
    }
    fn owns(&self, a: usize, end: usize) -> bool {
        let mut at = a;
        while at < end {
            let Some((_, r)) = self.find(at) else {
                return false;
            };
            at = r.end.min(end);
        }
        true
    }
    fn split(&mut self, a: usize) {
        if let Some((s, r)) = self
            .find(a)
            .filter(|(s, _)| *s < a)
            .map(|(s, r)| (s, r.clone()))
        {
            self.regions.get_mut(&s).unwrap().end = a;
            self.regions.insert(a, r);
        }
    }
    fn remove(&mut self, a: usize, e: usize) {
        self.split(a);
        self.split(e);
        self.regions.retain(|&s, _| s < a || s >= e);
        self.dirty.retain(|&s| s < a || s >= e);
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MEMORY_BASIC_INFORMATION {
    pub BaseAddress: *mut c_void,
    pub AllocationBase: *mut c_void,
    pub AllocationProtect: u32,
    pub PartitionId: u16,
    pub RegionSize: usize,
    pub State: u32,
    pub Protect: u32,
    pub Type: u32,
}
#[repr(C)]
pub struct MEMORY_MAPPED_VIEW_ADDRESS {
    pub Value: *mut c_void,
}
fn bounds(a: usize, n: usize) -> Option<(usize, usize)> {
    if n == 0 {
        return None;
    }
    let e = a.checked_add(n)?.checked_add(PAGE - 1)? & !(PAGE - 1);
    Some((a & !(PAGE - 1), e))
}
fn failure() -> *mut c_void {
    std::ptr::null_mut()
}
pub unsafe fn GetLastError() -> u32 {
    std::io::Error::last_os_error()
        .raw_os_error()
        .unwrap_or(libc::EINVAL) as u32
}
pub unsafe fn VirtualAlloc(
    address: *const c_void,
    size: usize,
    flags: u32,
    protection: u32,
) -> *mut c_void {
    if flags & MEM_WRITE_WATCH != 0 {
        if crate::platform::signals::ensure_installed().is_err() {
            return failure();
        }
    }
    let Some((a, e)) = bounds(address as usize, size) else {
        return failure();
    };
    let mut maps = MAPS.lock();
    let commit = flags & MEM_COMMIT != 0;
    if a != 0
        && maps.owns(a, e)
        && (flags & MEM_RESERVE == 0 || flags & MEM_REPLACE_PLACEHOLDER != 0)
    {
        maps.split(a);
        maps.split(e);
        if flags & MEM_REPLACE_PLACEHOLDER != 0 {
            for (_, r) in maps.regions.range_mut(a..e) {
                r.allocation = a;
                r.watch = flags & MEM_WRITE_WATCH != 0;
            }
        }
        for (&s, r) in maps.regions.range_mut(a..e) {
            let access = if commit {
                prot(protection) & if r.watch { !libc::PROT_WRITE } else { -1 }
            } else {
                0
            };
            if unsafe { libc::mprotect(s as *mut _, r.end - s, access) } != 0 {
                return failure();
            }
            r.committed = commit;
            r.protection = protection;
        }
        return a as *mut _;
    }
    let watch = flags & MEM_WRITE_WATCH != 0;
    let access = if commit {
        prot(protection) & if watch { !libc::PROT_WRITE } else { -1 }
    } else {
        0
    };
    let p = unsafe {
        libc::mmap(
            a as *mut _,
            e - a,
            access,
            libc::MAP_PRIVATE
                | libc::MAP_ANONYMOUS
                | libc::MAP_NORESERVE
                | if a != 0 { libc::MAP_FIXED_NOREPLACE } else { 0 },
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return failure();
    }
    if a != 0 && p as usize != a {
        unsafe {
            libc::munmap(p, e - a);
        }
        return failure();
    }
    let at = p as usize;
    maps.regions.insert(
        at,
        Region {
            end: at + e - a,
            allocation: at,
            protection,
            committed: commit,
            mapped: false,
            watch,
            section: None,
        },
    );
    p
}
pub unsafe fn VirtualAlloc2(
    _process: HANDLE,
    address: *const c_void,
    size: usize,
    flags: u32,
    protection: u32,
    _params: *const c_void,
    _count: u32,
) -> *mut c_void {
    unsafe { VirtualAlloc(address, size, flags, protection) }
}
pub unsafe fn VirtualFree(address: *mut c_void, size: usize, flags: u32) -> i32 {
    let a = address as usize;
    let mut maps = MAPS.lock();
    let Some((_, r)) = maps.find(a).map(|(s, r)| (s, r.clone())) else {
        return 0;
    };
    let end = if size == 0 {
        maps.regions
            .values()
            .filter(|v| v.allocation == r.allocation)
            .map(|r| r.end)
            .max()
            .unwrap_or(r.end)
    } else {
        let Some((_, e)) = bounds(a, size) else {
            return 0;
        };
        e
    };
    if !maps.owns(a, end) {
        return 0;
    }
    let r_allocation = r.allocation;
    if flags & MEM_PRESERVE_PLACEHOLDER != 0
        && maps
            .regions
            .range(a..end)
            .any(|(_, r)| r.committed || r.watch)
    {
        // Releasing private backing preserves the address reservation, but discards contents
        // and write-watch state. A split of an untouched placeholder needs no syscall.
        if r.mapped {
            return 0;
        }
        let p = unsafe {
            libc::mmap(
                address,
                end - a,
                0,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return 0;
        }
        maps.remove(a, end);
        maps.regions.insert(
            a,
            Region {
                end,
                allocation: a,
                protection: PAGE_NOACCESS,
                committed: false,
                mapped: false,
                watch: false,
                section: None,
            },
        );
        return 1;
    }
    if flags & MEM_PRESERVE_PLACEHOLDER != 0 || flags & MEM_COALESCE_PLACEHOLDERS != 0 {
        // Splitting/coalescing a placeholder only changes bookkeeping. Already inaccessible.
        maps.split(a);
        maps.split(end);
        if flags & MEM_COALESCE_PLACEHOLDERS != 0 {
            if maps
                .regions
                .range(a..end)
                .any(|(_, r)| r.committed || r.mapped)
            {
                return 0;
            }
            maps.remove(a, end);
            maps.regions.insert(
                a,
                Region {
                    end,
                    allocation: a,
                    protection: PAGE_NOACCESS,
                    committed: false,
                    mapped: false,
                    watch: false,
                    section: None,
                },
            );
        } else {
            for (&s, r) in maps.regions.iter_mut() {
                if !r.committed && r.allocation == r_allocation {
                    r.allocation = s;
                }
            }
        }
        return 1;
    }
    if flags & MEM_DECOMMIT != 0 {
        if unsafe { libc::mprotect(address, end - a, 0) } != 0 {
            return 0;
        }
        if unsafe { libc::madvise(address, end - a, libc::MADV_DONTNEED) } != 0 {
            return 0;
        }
        maps.split(a);
        maps.split(end);
        for (_, r) in maps.regions.range_mut(a..end) {
            r.committed = false;
            r.protection = PAGE_NOACCESS;
        }
        maps.dirty.retain(|&p| p < a || p >= end);
        return 1;
    }
    if flags & MEM_RELEASE != 0 && unsafe { libc::munmap(address, end - a) } == 0 {
        maps.remove(a, end);
        1
    } else {
        0
    }
}
pub unsafe fn VirtualProtect(
    address: *const c_void,
    size: usize,
    protection: u32,
    old: *mut u32,
) -> i32 {
    let Some((a, e)) = bounds(address as usize, size) else {
        return 0;
    };
    let mut maps = MAPS.lock();
    if !maps.owns(a, e) {
        if !old.is_null() {
            unsafe {
                *old = PAGE_READWRITE;
            }
        }
        return (unsafe { libc::mprotect(a as *mut _, e - a, prot(protection)) } == 0) as i32;
    }
    if !old.is_null() {
        unsafe {
            *old = maps.find(a).unwrap().1.protection;
        }
    }
    maps.split(a);
    maps.split(e);
    for (&s, r) in maps.regions.range_mut(a..e) {
        if unsafe {
            libc::mprotect(
                s as *mut _,
                r.end - s,
                prot(protection) & if r.watch { !libc::PROT_WRITE } else { -1 },
            )
        } != 0
        {
            return 0;
        }
        r.protection = protection;
    }
    1
}
pub unsafe fn VirtualQuery(
    address: *const c_void,
    out: *mut MEMORY_BASIC_INFORMATION,
    _size: usize,
) -> usize {
    if out.is_null() {
        return 0;
    }
    let a = address as usize;
    let maps = MAPS.lock();
    if let Some((s, r)) = maps.find(a) {
        let (end, armed) = if r.committed && r.protection == PAGE_READWRITE {
            crate::gpu::resident::linux_protection_span(a as u64, r.end as u64)
        } else { (r.end as u64, false) };
        let start = if r.committed && r.protection == PAGE_READWRITE { a & !(PAGE - 1) } else { s };
        unsafe {
            *out = MEMORY_BASIC_INFORMATION {
                BaseAddress: start as _,
                AllocationBase: r.allocation as _,
                AllocationProtect: r.protection,
                PartitionId: 0,
                RegionSize: end as usize - start,
                State: if r.committed { MEM_COMMIT } else { MEM_RESERVE },
                Protect: if armed { PAGE_READONLY } else { r.protection },
                Type: if r.mapped { MEM_MAPPED } else { MEM_PRIVATE },
            };
        }
        return std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
    }
    drop(maps);
    // Includes Rust stacks/heap and Vulkan driver mappings; queried only outside signal context.
    if let Ok(text) = std::fs::read_to_string("/proc/self/maps") {
        let mut next = usize::MAX;
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let Some(range) = fields.next() else { continue };
            let Some((lo, hi)) = range.split_once('-') else {
                continue;
            };
            let (Ok(s), Ok(e)) = (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16))
            else {
                continue;
            };
            if s > a {
                next = next.min(s);
            }
            if a >= s && a < e {
                let p = fields.next().unwrap_or("---").as_bytes();
                let protection = match (
                    p.first() == Some(&b'r'),
                    p.get(1) == Some(&b'w'),
                    p.get(2) == Some(&b'x'),
                ) {
                    (_, true, true) => PAGE_EXECUTE_READWRITE,
                    (_, true, false) => PAGE_READWRITE,
                    (true, false, true) => PAGE_EXECUTE_READ,
                    (true, false, false) => PAGE_READONLY,
                    (_, _, true) => PAGE_EXECUTE,
                    _ => PAGE_NOACCESS,
                };
                unsafe {
                    *out = MEMORY_BASIC_INFORMATION {
                        BaseAddress: s as _,
                        AllocationBase: s as _,
                        AllocationProtect: protection,
                        PartitionId: 0,
                        RegionSize: e - s,
                        State: MEM_COMMIT,
                        Protect: protection,
                        Type: MEM_PRIVATE,
                    };
                }
                return std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
            }
        }
        if next != usize::MAX {
            let s = a & !(PAGE - 1);
            unsafe {
                *out = MEMORY_BASIC_INFORMATION {
                    BaseAddress: s as _,
                    AllocationBase: std::ptr::null_mut(),
                    AllocationProtect: 0,
                    PartitionId: 0,
                    RegionSize: next - s,
                    State: MEM_FREE,
                    Protect: PAGE_NOACCESS,
                    Type: 0,
                };
            }
            return std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        }
    }
    0
}
/// udmabuf_create from linux/udmabuf.h: _IOW('u', 0x42, 24-byte struct).
fn section_to_dmabuf(fd: i32, offset: u64, size: u64) -> Option<i32> {
    #[repr(C)]
    struct UdmabufCreate { memfd: u32, flags: u32, offset: u64, size: u64 }
    const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
    let dev = unsafe { libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if dev < 0 {
        DMA_IOCTL_FAIL.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let c = UdmabufCreate { memfd: fd as u32, flags: 0, offset, size };
    let r = unsafe { libc::ioctl(dev, UDMABUF_CREATE, &c) };
    let err = std::io::Error::last_os_error();
    unsafe { libc::close(dev) };
    if r < 0 {
        DMA_IOCTL_FAIL.fetch_add(1, Ordering::Relaxed);
        crate::warn!("udmabuf ioctl fd={fd} off={offset:#x} size={size:#x}: {err}");
    }
    (r >= 0).then_some(r)
}

use std::sync::atomic::{AtomicU64, Ordering};
static HOST_OK: AtomicU64 = AtomicU64::new(0);
static HOST_FAIL: AtomicU64 = AtomicU64::new(0);
static DMA_OK: AtomicU64 = AtomicU64::new(0);
static DMA_FAIL: AtomicU64 = AtomicU64::new(0);
static DMA_NO_SECTION: AtomicU64 = AtomicU64::new(0);
static DMA_CROSSES_MEMFD: AtomicU64 = AtomicU64::new(0);
static DMA_UNALIGNED: AtomicU64 = AtomicU64::new(0);
static DMA_IOCTL_FAIL: AtomicU64 = AtomicU64::new(0);

pub fn note_host(ok: bool) {
    if ok { HOST_OK.fetch_add(1, Ordering::Relaxed); } else { HOST_FAIL.fetch_add(1, Ordering::Relaxed); }
}
pub fn note_dma(ok: bool) {
    if ok { DMA_OK.fetch_add(1, Ordering::Relaxed); } else { DMA_FAIL.fetch_add(1, Ordering::Relaxed); }
}

pub fn dma_status() -> String {
    format!("    dma-buf imports: host ok {}, host failed {}, dma ok {}, dma import failed {}, no section {}, crosses memfd {}, unaligned {}, ioctl failed {}\n",
        HOST_OK.load(Ordering::Relaxed), HOST_FAIL.load(Ordering::Relaxed),
        DMA_OK.load(Ordering::Relaxed), DMA_FAIL.load(Ordering::Relaxed),
        DMA_NO_SECTION.load(Ordering::Relaxed), DMA_CROSSES_MEMFD.load(Ordering::Relaxed),
        DMA_UNALIGNED.load(Ordering::Relaxed), DMA_IOCTL_FAIL.load(Ordering::Relaxed))
}

/// The dma-buf and the exact range it covers for a host pointer inside a
/// section. Views are windows onto one large memfd, so the offset is absolute;
/// a range past the memfd's end or not page-aligned is refused: the import's
/// length is what vkCreateBuffer sees, and a shorter dma-buf would let the GPU
/// write outside the buffer and lose the device.
pub fn host_import_source(ptr: u64, size: u64) -> Option<(i32, u64)> {
    let maps = MAPS.lock();
    let at = ptr as usize;
    let Some((start, r)) = maps.find(at) else {
        DMA_NO_SECTION.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let Some((fd, base)) = r.section else {
        DMA_NO_SECTION.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let off = base + (at - start) as u64;
    const PAGE: u64 = 4096;
    if off % PAGE != 0 || size % PAGE != 0 {
        DMA_UNALIGNED.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if size == 0 || off.checked_add(size).is_none() || unsafe { libc::fstat(fd, &mut st) } != 0 || off + size > st.st_size as u64 {
        DMA_CROSSES_MEMFD.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    section_to_dmabuf(fd, off, size).map(|d| (d, size))
}

pub unsafe fn CreateFileMappingW(
    _file: HANDLE,
    _attrs: *const c_void,
    _protect: u32,
    hi: u32,
    lo: u32,
    _name: *const u16,
) -> HANDLE {
    let fd = unsafe {
        libc::memfd_create(c"regaliant-section".as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
    };
    if fd < 0 {
        return failure();
    }
    let len = ((hi as u64) << 32) | lo as u64;
    if unsafe { libc::ftruncate(fd, len as i64) } != 0 {
        unsafe {
            libc::close(fd);
        }
        return failure();
    }
    unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) };
    (fd as usize + 1) as HANDLE
}
pub unsafe fn MapViewOfFile3(
    section: HANDLE,
    _process: HANDLE,
    address: *const c_void,
    off: u64,
    size: usize,
    flags: u32,
    protection: u32,
    _params: *const c_void,
    _count: u32,
) -> MEMORY_MAPPED_VIEW_ADDRESS {
    let mut maps = MAPS.lock();
    let a = address as usize;
    let Some((_, e)) = bounds(a, size) else {
        return MEMORY_MAPPED_VIEW_ADDRESS { Value: failure() };
    };
    let fixed = if a == 0 {
        0
    } else if flags & MEM_REPLACE_PLACEHOLDER != 0 && maps.owns(a, e) {
        libc::MAP_FIXED
    } else {
        libc::MAP_FIXED_NOREPLACE
    };
    let p = unsafe {
        libc::mmap(
            a as _,
            size,
            prot(protection),
            libc::MAP_SHARED | fixed,
            section as i32 - 1,
            off as i64,
        )
    };
    if p == libc::MAP_FAILED {
        return MEMORY_MAPPED_VIEW_ADDRESS { Value: failure() };
    }
    let s = p as usize;
    maps.remove(s, s + size);
    maps.regions.insert(
        s,
        Region {
            end: s + size,
            allocation: s,
            protection,
            committed: true,
            mapped: true,
            watch: false,
            section: Some((section as i32 - 1, off)),
        },
    );
    MEMORY_MAPPED_VIEW_ADDRESS { Value: p }
}
pub unsafe fn UnmapViewOfFile2(
    _process: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    flags: u32,
) -> i32 {
    let mut maps = MAPS.lock();
    let a = view.Value as usize;
    let Some((s, mut r)) = maps.find(a).map(|(s, r)| (s, r.clone())) else {
        return 0;
    };
    if a != s {
        return 0;
    }
    r.end = maps
        .regions
        .values()
        .filter(|v| v.allocation == r.allocation)
        .map(|v| v.end)
        .max()
        .unwrap_or(r.end);
    if flags & MEM_PRESERVE_PLACEHOLDER != 0 {
        let p = unsafe {
            libc::mmap(
                view.Value,
                r.end - a,
                0,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return 0;
        }
        maps.remove(a, r.end);
        maps.regions.insert(
            a,
            Region {
                protection: PAGE_NOACCESS,
                committed: false,
                mapped: false,
                ..r
            },
        );
        1
    } else {
        if unsafe { libc::munmap(view.Value, r.end - a) } != 0 {
            return 0;
        }
        maps.remove(a, r.end);
        1
    }
}
/// A queued fault may outlive the temporary no-access interval of a section split. Only
/// retry when our owned, committed mapping currently permits the actual access. Unowned
/// addresses, guard pages and genuine protection violations remain fatal.
pub fn mapping_allows(address: usize, write: bool) -> bool {
    let maps = MAPS.lock();
    maps.find(address).is_some_and(|(_, r)| {
        r.committed
            && r.protection & PAGE_GUARD == 0
            && prot(r.protection)
                & if write {
                    libc::PROT_WRITE
                } else {
                    libc::PROT_READ
                }
                != 0
            && !(write && r.watch)
    })
}

/// Called by the fault service, never directly from the signal handler.
pub fn write_watch_fault(address: usize) -> bool {
    let page = address & !(PAGE - 1);
    let mut maps = MAPS.lock();
    let Some((_, r)) = maps.find(page) else {
        return false;
    };
    if !r.watch || !r.committed || !writable(r.protection) {
        return false;
    }
    let access = prot(r.protection);
    if unsafe { libc::mprotect(page as _, PAGE, access) } != 0 {
        return false;
    }
    maps.dirty.insert(page);
    true
}
pub unsafe fn GetWriteWatch(
    flags: u32,
    address: *const c_void,
    size: usize,
    pages: *mut *mut c_void,
    count: *mut usize,
    granularity: *mut u32,
) -> u32 {
    if count.is_null() || granularity.is_null() {
        return libc::EINVAL as u32;
    }
    let Some((a, e)) = bounds(address as usize, size) else {
        return libc::EINVAL as u32;
    };
    let mut maps = MAPS.lock();
    let Some((_, r)) = maps.find(a) else {
        return libc::EINVAL as u32;
    };
    if !r.watch || !maps.owns(a, e) {
        return libc::EINVAL as u32;
    }
    let dirty: Vec<_> = maps.dirty.range(a..e).copied().collect();
    let cap = unsafe { *count };
    unsafe {
        *granularity = PAGE as u32;
        *count = dirty.len();
    }
    if dirty.len() > cap {
        return libc::ENOSPC as u32;
    }
    if pages.is_null() && !dirty.is_empty() {
        return libc::EINVAL as u32;
    }
    for (i, p) in dirty.iter().enumerate() {
        unsafe {
            *pages.add(i) = *p as _;
        }
    }
    if flags & 1 != 0 {
        maps.split(a);
        maps.split(e);
        for (&s, r) in maps.regions.range(a..e) {
            if r.committed
                && unsafe {
                    libc::mprotect(s as _, r.end - s, prot(r.protection) & !libc::PROT_WRITE)
                } != 0
            {
                return unsafe { GetLastError() };
            }
        }
        maps.dirty.retain(|&p| p < a || p >= e);
    }
    0
}
pub unsafe fn CloseHandle(h: HANDLE) -> i32 {
    if h as isize > 0 {
        (unsafe { libc::close(h as i32 - 1) } == 0) as i32
    } else {
        1
    }
}
pub unsafe fn GetCurrentProcess() -> HANDLE {
    INVALID_HANDLE_VALUE
}
pub unsafe fn GetCurrentProcessId() -> u32 {
    std::process::id()
}
pub unsafe fn GetCurrentThreadId() -> u32 {
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}
pub unsafe fn GetCurrentThread() -> HANDLE {
    unsafe { (-(GetCurrentThreadId() as isize) - 2) as HANDLE }
}
pub unsafe fn GetCurrentProcessorNumber() -> u32 {
    unsafe { libc::sched_getcpu().max(0) as u32 }
}
pub unsafe fn QueryPerformanceCounter(out: *mut i64) -> i32 {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } != 0 {
        return 0;
    }
    unsafe {
        *out = ts.tv_sec * 1_000_000_000 + ts.tv_nsec;
    }
    1
}
pub unsafe fn QueryPerformanceFrequency(out: *mut i64) -> i32 {
    unsafe {
        *out = 1_000_000_000;
    }
    1
}
pub const INFINITE: u32 = u32::MAX;
pub unsafe fn WaitOnAddress(
    address: *const c_void,
    compare: *const c_void,
    size: usize,
    ms: u32,
) -> i32 {
    static RETURN_ON_WAKE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let return_on_wake = *RETURN_ON_WAKE.get_or_init(|| std::env::var("DS_LINUX_ADDRESS_WAKE").ok().as_deref() != Some("0"));
    unsafe { wait_on_address(address, compare, size, ms, return_on_wake) }
}

unsafe fn wait_on_address(address: *const c_void, compare: *const c_void, size: usize, ms: u32, return_on_wake: bool) -> i32 {
    if !matches!(size, 1 | 2 | 4 | 8) {
        return 0;
    }
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_millis(if ms == INFINITE { 86_400_000 } else { ms } as u64);
    loop {
        if unsafe { libc::memcmp(address, compare, size) } != 0 {
            return 1;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return 0;
        }
        // Futex operates on the low 32 bits; a wake accompanies changes to either half.
        let timeout = (deadline - now).min(std::time::Duration::from_millis(10));
        let ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: timeout.as_nanos() as _,
        };
        if size >= 4 && address as usize % 4 == 0 {
            // Compare against the caller's snapshot, not a newly loaded word that could
            // have changed since memcmp. EAGAIN is a valid early wake too.
            let seen = unsafe { if return_on_wake { std::ptr::read_unaligned(compare as *const u32) } else { std::ptr::read_volatile(address as *const u32) } };
            let result = unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    address,
                    libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                    seen,
                    &ts,
                )
            };
            // A wake can publish a GPU label promise without writing the guest word.
            // The caller rechecks both conditions. Hiding that wake here forced a full
            // 1ms timeout per cross-queue wait. Do not weaken any renderer dependency.
            if return_on_wake && (result == 0 || unsafe { *libc::__errno_location() } == libc::EAGAIN) {
                return 1;
            }
        } else {
            std::thread::sleep(timeout);
        }
    }
}
pub unsafe fn WakeByAddressAll(address: *const c_void) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            address,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        );
    }
}
pub const THREAD_PRIORITY_NORMAL: i32 = 0;
pub const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
pub const THREAD_PRIORITY_LOWEST: i32 = -2;
pub const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
pub const THREAD_PRIORITY_HIGHEST: i32 = 2;
pub const THREAD_QUERY_LIMITED_INFORMATION: u32 = 0;
pub const THREAD_SET_LIMITED_INFORMATION: u32 = 0;
pub const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0;
pub unsafe fn SetThreadPriority(_h: HANDLE, _priority: i32) -> i32 {
    unsafe {
        *libc::__errno_location() = libc::ENOTSUP;
    }
    0
} // No silent realtime elevation.
pub unsafe fn OpenThread(_access: u32, _inherit: i32, id: u32) -> HANDLE {
    (-(id as isize) - 2) as HANDLE
}
pub unsafe fn OpenProcess(_access: u32, _inherit: i32, id: u32) -> HANDLE {
    (-(id as isize) - 2) as HANDLE
}
pub unsafe fn GetExitCodeProcess(h: HANDLE, out: *mut u32) -> i32 {
    unsafe {
        *out = if libc::kill((-(h as isize) - 2) as i32, 0) == 0 {
            259
        } else {
            0
        };
    }
    1
}
pub unsafe fn ReadProcessMemory(
    _h: HANDLE,
    base: *const c_void,
    to: *mut c_void,
    size: usize,
    read: *mut usize,
) -> i32 {
    let local = libc::iovec {
        iov_base: to,
        iov_len: size,
    };
    let remote = libc::iovec {
        iov_base: base as _,
        iov_len: size,
    };
    let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
    if !read.is_null() {
        unsafe {
            *read = n.max(0) as usize;
        }
    }
    (n == size as isize) as i32
}
pub unsafe fn GetEnvironmentVariableW(name: *const u16, out: *mut u16, size: u32) -> u32 {
    let mut n = 0;
    while unsafe { *name.add(n) } != 0 {
        n += 1;
    }
    let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(name, n) });
    let Ok(value) = std::env::var(name) else {
        return 0;
    };
    let w: Vec<_> = value.encode_utf16().chain(Some(0)).collect();
    if w.len() > size as usize {
        return w.len() as u32;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(w.as_ptr(), out, w.len());
    }
    (w.len() - 1) as u32
}
#[repr(C)]
#[derive(Default)]
pub struct MEMORYSTATUSEX {
    pub dwLength: u32,
    pub dwMemoryLoad: u32,
    pub ullTotalPhys: u64,
    pub ullAvailPhys: u64,
    pub ullTotalPageFile: u64,
    pub ullAvailPageFile: u64,
    pub ullTotalVirtual: u64,
    pub ullAvailVirtual: u64,
    pub ullAvailExtendedVirtual: u64,
}
pub unsafe fn GlobalMemoryStatusEx(out: *mut MEMORYSTATUSEX) -> i32 {
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    if unsafe { libc::sysinfo(&mut info) } != 0 {
        return 0;
    }
    let u = info.mem_unit as u64;
    unsafe {
        (*out).ullTotalPhys = info.totalram * u;
        (*out).ullAvailPhys = info.freeram * u;
        (*out).ullTotalPageFile = (info.totalram + info.totalswap) * u;
        (*out).ullAvailPageFile = (info.freeram + info.freeswap) * u;
    }
    1
}
#[repr(C)]
#[derive(Default)]
pub struct PROCESS_MEMORY_COUNTERS {
    pub cb: u32,
    pub PageFaultCount: u32,
    pub PeakWorkingSetSize: usize,
    pub WorkingSetSize: usize,
    pub QuotaPeakPagedPoolUsage: usize,
    pub QuotaPagedPoolUsage: usize,
    pub QuotaPeakNonPagedPoolUsage: usize,
    pub QuotaNonPagedPoolUsage: usize,
    pub PagefileUsage: usize,
    pub PeakPagefileUsage: usize,
}
#[repr(C)]
pub struct PROCESS_MEMORY_COUNTERS_EX {
    pub base: PROCESS_MEMORY_COUNTERS,
    pub PrivateUsage: usize,
}
pub unsafe fn GetProcessMemoryInfo(
    _h: HANDLE,
    out: *mut PROCESS_MEMORY_COUNTERS,
    _size: u32,
) -> i32 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) } != 0 {
        return 0;
    }
    unsafe {
        (*out).PageFaultCount = (u.ru_minflt + u.ru_majflt) as u32;
        (*out).PeakWorkingSetSize = u.ru_maxrss as usize * 1024;
    }
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        let v: Vec<usize> = s
            .split_whitespace()
            .filter_map(|v| v.parse().ok())
            .collect();
        if v.len() > 1 {
            unsafe {
                (*out).WorkingSetSize = v[1] * PAGE;
                (*out).PagefileUsage = v[0] * PAGE;
            }
        }
    }
    1
}
pub mod Win32 {
    pub mod Foundation {
        pub use super::super::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE};
    }
    pub mod Storage {
        pub mod FileSystem {
            pub use super::super::super::GetDiskFreeSpaceExW;
        }
    }
    pub mod System {
        pub mod Memory {
            pub use super::super::super::*;
        }
        pub mod Threading {
            pub use super::super::super::*;
        }
        pub mod Performance {
            pub use super::super::super::{QueryPerformanceCounter, QueryPerformanceFrequency};
        }
        pub mod Environment {
            pub use super::super::super::GetEnvironmentVariableW;
        }
        pub mod SystemInformation {
            pub use super::super::super::{
                GetLogicalProcessorInformationEx, GlobalMemoryStatusEx, MEMORYSTATUSEX,
                RelationCache, RelationProcessorCore,
            };
        }
        pub mod ProcessStatus {
            pub use super::super::super::{
                GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
            };
        }
        pub mod WindowsProgramming {
            pub use super::super::super::QueryThreadCycleTime;
        }
        pub mod Diagnostics {
            pub mod Debug {
                pub use super::super::super::super::{
                    CONTEXT, GetThreadContext, ReadProcessMemory, RtlCaptureStackBackTrace,
                };
            }
            pub mod ToolHelp {
                pub use super::super::super::super::{
                    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First,
                    Thread32Next,
                };
            }
        }
    }
}

pub trait OsStrExt {
    fn encode_wide(&self) -> std::vec::IntoIter<u16>;
}
impl OsStrExt for std::ffi::OsStr {
    fn encode_wide(&self) -> std::vec::IntoIter<u16> {
        self.to_string_lossy()
            .encode_utf16()
            .collect::<Vec<_>>()
            .into_iter()
    }
}
pub unsafe fn GetDiskFreeSpaceExW(
    path: *const u16,
    free: *mut u64,
    total: *mut u64,
    available: *mut u64,
) -> i32 {
    let mut n = 0;
    while unsafe { *path.add(n) } != 0 {
        n += 1;
    }
    let s = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(path, n) });
    let Ok(p) = std::ffi::CString::new(s) else {
        return 0;
    };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(p.as_ptr(), &mut st) } != 0 {
        return 0;
    }
    unsafe {
        if !free.is_null() {
            *free = st.f_bavail * st.f_frsize;
        }
        if !total.is_null() {
            *total = st.f_blocks * st.f_frsize;
        }
        if !available.is_null() {
            *available = st.f_bfree * st.f_frsize;
        }
    }
    1
}
pub unsafe fn GetProcessAffinityMask(_h: HANDLE, process: *mut usize, system: *mut usize) -> i32 {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) } != 0 {
        return 0;
    }
    let mut mask = 0;
    for cpu in 0..64 {
        if unsafe { libc::CPU_ISSET(cpu, &set) } {
            mask |= 1usize << cpu;
        }
    }
    unsafe {
        *process = mask;
        *system = mask;
    }
    1
}
pub unsafe fn SetThreadAffinityMask(h: HANDLE, mask: usize) -> usize {
    static ALLOWED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let allowed = *ALLOWED.get_or_init(|| {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        if unsafe { libc::sched_getaffinity(libc::getpid(), std::mem::size_of_val(&set), &mut set) } != 0 { return 0; }
        (0..64).filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) }).fold(0usize, |mask, cpu| mask | (1usize << cpu))
    });
    let mask = mask & allowed;
    if mask == 0 { return 0; }

    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for cpu in 0..64 {
        if mask & (1 << cpu) != 0 {
            unsafe {
                libc::CPU_SET(cpu, &mut set);
            }
        }
    }
    if unsafe {
        libc::sched_setaffinity(
            (-(h as isize) - 2) as i32,
            std::mem::size_of_val(&set),
            &set,
        )
    } == 0
    {
        mask
    } else {
        0
    }
}
pub unsafe fn QueryThreadCycleTime(_h: HANDLE, _cycles: *mut u64) -> i32 {
    0
}
pub const RelationProcessorCore: i32 = 0;
pub const RelationCache: i32 = 2;
pub unsafe fn GetLogicalProcessorInformationEx(
    _relation: i32,
    _buffer: *mut c_void,
    _len: *mut u32,
) -> i32 {
    0
}
pub const THREAD_GET_CONTEXT: u32 = 0;
pub const THREAD_SUSPEND_RESUME: u32 = 0;
pub const TH32CS_SNAPTHREAD: u32 = 4;
#[repr(C)]
pub struct THREADENTRY32 {
    pub dwSize: u32,
    pub cntUsage: u32,
    pub th32ThreadID: u32,
    pub th32OwnerProcessID: u32,
    pub tpBasePri: i32,
    pub tpDeltaPri: i32,
    pub dwFlags: u32,
}
pub unsafe fn CreateToolhelp32Snapshot(_flags: u32, _pid: u32) -> HANDLE {
    INVALID_HANDLE_VALUE
}
pub unsafe fn Thread32First(_h: HANDLE, _entry: *mut THREADENTRY32) -> i32 {
    0
}
pub unsafe fn Thread32Next(_h: HANDLE, _entry: *mut THREADENTRY32) -> i32 {
    0
}
pub unsafe fn SuspendThread(_h: HANDLE) -> u32 {
    u32::MAX
}
impl std::ops::Deref for PROCESS_MEMORY_COUNTERS_EX {
    type Target = PROCESS_MEMORY_COUNTERS;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl std::ops::DerefMut for PROCESS_MEMORY_COUNTERS_EX {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

#[repr(C)]
pub struct CONTEXT {
    pub ContextFlags: u32,
}
pub unsafe fn GetThreadContext(_h: HANDLE, _ctx: *mut CONTEXT) -> i32 {
    0
}
pub unsafe fn ResumeThread(_h: HANDLE) -> u32 {
    u32::MAX
}
pub unsafe fn RtlCaptureStackBackTrace(
    skip: u32,
    count: u32,
    out: *mut *mut c_void,
    _hash: *mut u32,
) -> u16 {
    unsafe extern "C" {
        fn backtrace(buffer: *mut *mut c_void, size: i32) -> i32;
    }
    let mut frames = [std::ptr::null_mut(); 128];
    let n = unsafe { backtrace(frames.as_mut_ptr(), 128) }.max(0) as usize;
    let skip = (skip as usize).min(n);
    let take = (count as usize).min(n - skip);
    unsafe {
        std::ptr::copy_nonoverlapping(frames.as_ptr().add(skip), out, take);
    }
    take as u16
}

/// Host I/O cannot take a userspace write fault: explicitly dirty/unprotect watched buffers.
pub fn prepare_kernel_write(base: u64, len: u64) {
    let Some(end) = base.checked_add(len) else {
        return;
    };
    let mut page = base as usize & !(PAGE - 1);
    while page < (end as usize) {
        write_watch_fault(page);
        page += PAGE;
    }
}

#[cfg(test)]
mod tests {
    // Use a kernel-confirmed wake, not a scheduler-sensitive sleep before waking.
    fn wake_unchanged_word(return_on_wake: bool) -> i32 {
        use std::sync::{Arc, atomic::AtomicU64};
        let word = Arc::new(AtomicU64::new(7));
        let other = word.clone();
        let waiter = std::thread::spawn(move || {
            let expected = 7u64;
            unsafe { super::wait_on_address(other.as_ptr() as _, &expected as *const _ as _, 8, 250, return_on_wake) }
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let woken = unsafe { libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG, i32::MAX) };
            if woken > 0 { break; }
            assert!(!waiter.is_finished() && std::time::Instant::now() < deadline, "waiter never slept");
            std::thread::yield_now();
        }
        waiter.join().unwrap()
    }

    #[test]
    fn address_wake_returns_to_check_external_predicate() {
        // The old shim consumes the wake and eventually times out with unchanged memory.
        assert_eq!(wake_unchanged_word(false), 0);
        assert_eq!(wake_unchanged_word(true), 1);
    }

    #[test]
    fn address_wait_preserves_changed_value_and_timeout() {
        let word = 7u64;
        let changed = 6u64;
        unsafe {
            assert_eq!(super::wait_on_address(&word as *const _ as _, &changed as *const _ as _, 8, 0, true), 1);
            assert_eq!(super::wait_on_address(&word as *const _ as _, &word as *const _ as _, 8, 0, true), 0);
            assert_eq!(super::wait_on_address(&word as *const _ as _, &word as *const _ as _, 3, 0, true), 0);
        }
    }

    use super::*;

    #[test]
    fn positional_file_read_completes_after_short_copy_and_efault() {
        use std::os::fd::{FromRawFd, AsRawFd};
        use std::os::unix::fs::FileExt;
        crate::hle::kernel::time::init();
        unsafe {
            let fd = libc::memfd_create(c"regaliant-mixed-read-test".as_ptr(), libc::MFD_CLOEXEC);
            assert!(fd >= 0);
            let file = std::fs::File::from_raw_fd(fd);
            let bytes: Vec<u8> = (0..6 * PAGE + 73).map(|i| (i.wrapping_mul(37) ^ (i >> 9)) as u8).collect();
            assert_eq!(file.write_at(&bytes, 0).unwrap(), bytes.len());
            let p = VirtualAlloc(std::ptr::null(), 8 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
            assert!(!p.is_null());
            let _stack = crate::platform::signals::SignalStack::new().unwrap();
            std::ptr::write_bytes(p, 0xcc, 8 * PAGE);
            let reset = || {
                let mut pages = [std::ptr::null_mut(); 8];
                let mut n = pages.len();
                let mut granularity = 0;
                assert_eq!(GetWriteWatch(1, p, 8 * PAGE, pages.as_mut_ptr(), &mut n, &mut granularity), 0);
            };
            reset();
            // Mixed pages before entry, then force a re-arm between preparation and the
            // actual syscall. Linux returns a real short count first, then real EFAULT.
            assert!(write_watch_fault(p as usize));
            assert!(write_watch_fault(p as usize + 2 * PAGE));
            let out = std::slice::from_raw_parts_mut((p as *mut u8).add(17), bytes.len());
            let _write = crate::platform::file_hooks::host_write(out.as_ptr() as u64, out.len() as u64);
            let mut calls = 0;
            let mut saw_short = false;
            let mut saw_fault = false;
            let n = crate::mem::kernel_read_at(&|_| "mixed-protection file test".into(), out, 0, |buf, offset| {
                calls += 1;
                if calls <= 2 {
                    reset();
                    if calls == 1 { assert!(write_watch_fault(p as usize)); }
                }
                let r = file.read_at(buf, offset);
                if calls == 1 { saw_short = r.as_ref().is_ok_and(|&n| n > 0 && n < buf.len()); }
                if calls == 2 { saw_fault = r.as_ref().is_err_and(|e| e.raw_os_error() == Some(libc::EFAULT)); }
                r
            }).unwrap();
            assert!(saw_short && saw_fault && calls >= 3);
            assert_eq!(n, bytes.len());
            assert_eq!(out, bytes, "every byte, including offsets after partial copies");
            assert!(std::slice::from_raw_parts(p.cast::<u8>(), 17).iter().all(|&b| b == 0xcc));
            assert!(std::slice::from_raw_parts((p as *const u8).add(17 + n), 8 * PAGE - 17 - n).iter().all(|&b| b == 0xcc));
            assert!(file.as_raw_fd() >= 0);
            drop(_write);
            assert_eq!(VirtualFree(p, 0, MEM_RELEASE), 1);
        }
    }

    #[test]
    fn partial_decommit_zeros_only_recommitted_private_pages() {
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            assert!(!p.is_null());
            std::ptr::write_bytes(p, 0xff, 3 * PAGE);
            assert_eq!(VirtualAlloc(p, 3 * PAGE, MEM_COMMIT, PAGE_READWRITE), p);
            assert_eq!(*(p as *const u64), u64::MAX, "commit of committed memory preserves it");
            let middle = (p as *mut u8).add(PAGE);
            assert_eq!(VirtualFree(middle.cast(), PAGE, MEM_DECOMMIT), 1);
            assert_eq!(VirtualAlloc(middle.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE), middle.cast());
            assert!(std::slice::from_raw_parts(middle, PAGE).iter().all(|&x| x == 0));
            assert_eq!(*(p as *const u64), u64::MAX);
            assert_eq!(*(middle.add(PAGE) as *const u64), u64::MAX);
            assert_eq!(VirtualFree(p, 0, MEM_RELEASE), 1);
        }
    }

    #[test]
    fn write_watch_recovers_and_kernel_io_marks_dirty() {
        unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                8 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            let address = p as usize;
            // Actual concurrent SIGSEGVs, not simulated calls to the recovery routine.
            std::thread::scope(|scope| {
                for i in 0..8 {
                    scope.spawn(move || {
                        let _stack = crate::platform::signals::SignalStack::new().unwrap();
                        std::ptr::write_volatile((address + i * PAGE) as *mut u32, i as u32 + 1);
                    });
                }
            });
            let take = || {
                let mut pages = [std::ptr::null_mut(); 8];
                let (mut count, mut granularity) = (8, 0);
                assert_eq!(
                    GetWriteWatch(
                        1,
                        p,
                        8 * PAGE,
                        pages.as_mut_ptr(),
                        &mut count,
                        &mut granularity
                    ),
                    0
                );
                assert_eq!(granularity, 4096);
                pages[..count]
                    .iter()
                    .map(|v| *v as usize - address)
                    .collect::<Vec<_>>()
            };
            assert_eq!(take(), (0..8).map(|i| i * PAGE).collect::<Vec<_>>());
            assert!(take().is_empty());
            std::ptr::write_volatile((address + 3 * PAGE) as *mut u32, 99);
            assert_eq!(take(), vec![3 * PAGE]);
            let fd = libc::memfd_create(c"regaliant-io-test".as_ptr(), libc::MFD_CLOEXEC);
            assert!(fd >= 0);
            let bytes = [1u8, 2, 3, 4];
            assert_eq!(libc::write(fd, bytes.as_ptr().cast(), 4), 4);
            // Kernel copy_to_user returns EFAULT, without delivering SIGSEGV to this thread.
            assert_eq!(libc::pread(fd, p, 4, 0), -1);
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::EFAULT)
            );
            let n = crate::mem::kernel_io(
                &|| "Linux protected I/O test".into(),
                address as u64,
                4,
                true,
                || {
                    let n = libc::pread(fd, p, 4, 0);
                    if n < 0 {
                        Err(std::io::Error::last_os_error())
                    } else {
                        Ok(n)
                    }
                },
            )
            .unwrap();
            assert_eq!(n, 4);
            assert_eq!(std::slice::from_raw_parts(p.cast::<u8>(), 4), bytes);
            assert_eq!(take(), vec![0]);
            assert_eq!(libc::close(fd), 0);
            assert_eq!(VirtualFree(p, 0, MEM_RELEASE), 1);
        }
    }

    #[test]
    fn fixed_collision_never_replaces_existing_memory() {
        unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            std::ptr::write_volatile(p.cast::<u32>(), 0x12345678);
            assert!(VirtualAlloc(p, PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).is_null());
            assert_eq!(std::ptr::read_volatile(p.cast::<u32>()), 0x12345678);
            assert_eq!(VirtualFree(p, 0, MEM_RELEASE), 1);
        }
    }
}
