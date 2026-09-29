//! Issue #2313: the `map_async` completion callback must not panic when the
//! waiter has already returned and dropped its receiver.
//!
//! wgpu fires a pending map callback inline with `MapAborted` when the staging
//! buffer is dropped. On a timed-out wait the receivers drop before the staging
//! buffers, so the callback sends into a closed channel — during unwinding a
//! panic there aborts the host process.

use neat_ai_discovery::analysis::gpu::map_result_forwarder;
use std::sync::mpsc;

/// Reproduces the original trigger: the receiver is gone before wgpu delivers
/// the (aborted) mapping result.
#[test]
fn forwarder_does_not_panic_when_receiver_dropped() {
    let (sender, receiver) = mpsc::channel();
    let callback = map_result_forwarder(sender);
    drop(receiver);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback(Err(wgpu::BufferAsyncError));
    }));

    assert!(
        outcome.is_ok(),
        "map_async callback panicked after the waiter dropped its receiver"
    );
}

/// The live-waiter path still receives the mapping result unchanged.
#[test]
fn forwarder_delivers_result_to_live_receiver() {
    let (sender, receiver) = mpsc::channel();
    map_result_forwarder(sender)(Ok(()));
    assert!(matches!(receiver.try_recv(), Ok(Ok(()))));

    let (sender, receiver) = mpsc::channel();
    map_result_forwarder(sender)(Err(wgpu::BufferAsyncError));
    assert!(matches!(
        receiver.try_recv(),
        Ok(Err(wgpu::BufferAsyncError))
    ));
}
