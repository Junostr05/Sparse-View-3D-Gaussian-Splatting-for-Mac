import Darwin
import SplatIO

/// Number of SIGINT deliveries so far. Written only by the signal handler, read by the training watcher.
nonisolated(unsafe) var sparseSplatInterruptCount: Int = 0

/// Ctrl-C asks the run to stop: the watcher cancels training, which still stores its last completed model.
/// A second Ctrl-C ends the process at once, without storing anything.
func installInterruptHandler() {
    signal(SIGINT) { _ in
        if sparseSplatInterruptCount > 0 {
            _exit(CLIExitCode.interrupted)
        }
        sparseSplatInterruptCount += 1
    }
}

/// How many Ctrl-C presses arrived.
func interruptRequestCount() -> Int {
    sparseSplatInterruptCount
}
