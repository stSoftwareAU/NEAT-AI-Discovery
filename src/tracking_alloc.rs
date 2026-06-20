//! Inline tracking global allocator (Issue #1463).
//!
//! Replaces the unmaintained `cap` crate, which had been silent for ~39 months
//! and was wired in only for its allocated-bytes counter. [`TrackingAlloc`] is a
//! tiny `GlobalAlloc` wrapper around [`System`] that bumps an [`AtomicUsize`] on
//! every allocation and decrements it on every deallocation, exposing the live
//! total through [`TrackingAlloc::allocated`].
//!
//! The limit-enforcement behaviour of `cap` was never used here (the previous
//! allocator was constructed with a `usize::MAX` cap), so this wrapper tracks
//! only — it never refuses an allocation. Overhead is a single relaxed atomic
//! add/sub per allocation, which is negligible for polling every few seconds.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A [`GlobalAlloc`] wrapper around [`System`] that tracks the number of bytes
/// currently allocated through it.
pub struct TrackingAlloc {
    /// Live count of bytes handed out by `alloc`/`alloc_zeroed`/`realloc` and
    /// not yet returned via `dealloc`/`realloc`.
    allocated: AtomicUsize,
}

impl TrackingAlloc {
    /// Create a tracking allocator with a zeroed byte counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
        }
    }

    /// Return the number of bytes currently allocated through this allocator.
    ///
    /// This mirrors the `cap::Cap::allocated` API the crate previously relied on,
    /// so call sites need no change beyond the type swap.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

impl Default for TrackingAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every request is forwarded unchanged to the `System` allocator, which
// is itself a sound `GlobalAlloc`. The atomic bookkeeping only observes sizes
// that `System` already honours and never alters the returned pointers, so the
// safety contract of `GlobalAlloc` is upheld by delegation.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged to the System allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged to the System allocator.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the `ptr`/`layout` pair is forwarded unchanged to System and
        // matches the allocation that produced `ptr`, per the caller's contract.
        unsafe { System.dealloc(ptr, layout) };
        self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged to System,
        // matching the allocation that produced `ptr`, per the caller's contract.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // Apply the size delta. On failure (`new_ptr` null) the original
            // allocation is untouched, so the counter is left unchanged.
            let old_size = layout.size();
            if new_size >= old_size {
                self.allocated
                    .fetch_add(new_size - old_size, Ordering::Relaxed);
            } else {
                self.allocated
                    .fetch_sub(old_size - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_zero() {
        let alloc = TrackingAlloc::new();
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn default_starts_at_zero() {
        let alloc = TrackingAlloc::default();
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn alloc_then_dealloc_returns_to_zero() {
        let alloc = TrackingAlloc::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();

        // SAFETY: `layout` is valid (non-zero size) and the pointer returned is
        // freed exactly once below with the same layout.
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
        assert_eq!(alloc.allocated(), 1024);

        // SAFETY: `ptr` came from `alloc` above with the same `layout`.
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn alloc_zeroed_tracks_and_zeroes() {
        let alloc = TrackingAlloc::new();
        let layout = Layout::from_size_align(64, 8).unwrap();

        // SAFETY: valid layout; pointer freed once below with the same layout.
        let ptr = unsafe { alloc.alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        assert_eq!(alloc.allocated(), 64);

        // The returned memory must be zeroed.
        // SAFETY: `ptr` is valid for `layout.size()` bytes just allocated.
        let first_byte = unsafe { *ptr };
        assert_eq!(first_byte, 0);

        // SAFETY: `ptr` came from `alloc_zeroed` above with the same `layout`.
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn multiple_allocations_accumulate() {
        let alloc = TrackingAlloc::new();
        let layout = Layout::from_size_align(128, 8).unwrap();

        // SAFETY: valid layout; both pointers freed once below.
        let a = unsafe { alloc.alloc(layout) };
        // SAFETY: valid layout; freed once below.
        let b = unsafe { alloc.alloc(layout) };
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(alloc.allocated(), 256);

        // SAFETY: `a` came from `alloc` with `layout`.
        unsafe { alloc.dealloc(a, layout) };
        assert_eq!(alloc.allocated(), 128);
        // SAFETY: `b` came from `alloc` with `layout`.
        unsafe { alloc.dealloc(b, layout) };
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn realloc_grow_increases_counter() {
        let alloc = TrackingAlloc::new();
        let layout = Layout::from_size_align(100, 8).unwrap();

        // SAFETY: valid layout; result tracked and freed below.
        let ptr = unsafe { alloc.alloc(layout) };
        assert_eq!(alloc.allocated(), 100);

        // SAFETY: `ptr`/`layout` match the allocation above; grow to 300 bytes.
        let grown = unsafe { alloc.realloc(ptr, layout, 300) };
        assert!(!grown.is_null());
        assert_eq!(alloc.allocated(), 300);

        let grown_layout = Layout::from_size_align(300, 8).unwrap();
        // SAFETY: `grown` is the current allocation, freed once with its layout.
        unsafe { alloc.dealloc(grown, grown_layout) };
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn realloc_shrink_decreases_counter() {
        let alloc = TrackingAlloc::new();
        let layout = Layout::from_size_align(300, 8).unwrap();

        // SAFETY: valid layout; result tracked and freed below.
        let ptr = unsafe { alloc.alloc(layout) };
        assert_eq!(alloc.allocated(), 300);

        // SAFETY: `ptr`/`layout` match the allocation above; shrink to 50 bytes.
        let shrunk = unsafe { alloc.realloc(ptr, layout, 50) };
        assert!(!shrunk.is_null());
        assert_eq!(alloc.allocated(), 50);

        let shrunk_layout = Layout::from_size_align(50, 8).unwrap();
        // SAFETY: `shrunk` is the current allocation, freed once with its layout.
        unsafe { alloc.dealloc(shrunk, shrunk_layout) };
        assert_eq!(alloc.allocated(), 0);
    }

    /// The crate-wide global allocator is this type, so live Rust allocations
    /// during the test binary must be reflected by a non-zero counter.
    #[test]
    fn global_allocator_reports_live_usage() {
        let before = crate::ALLOCATOR.allocated();
        // Force a heap allocation that outlives the read below.
        let buf: Vec<u8> = vec![7u8; 4096];
        let after = crate::ALLOCATOR.allocated();
        assert!(after >= before + buf.len());
        // Keep `buf` alive until after the measurement.
        assert_eq!(buf[0], 7);
    }
}
