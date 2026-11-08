import SplatCore
import SwiftUI

/// The training queue, shown in the library window in place of the project grid. Jobs run in list order; a waiting
/// job's mode and preset can be changed, and the list can be reordered, edited and cleared.
struct QueueView: View {
    let queue: TrainingQueue
    @State private var selection: Set<UUID> = []

    var body: some View {
        Group {
            if queue.jobs.isEmpty {
                ContentUnavailableView {
                    Label("Queue Is Empty", systemImage: "list.number")
                } description: {
                    Text("Choose Add to Queue for projects in the library, or in a project's command palette.")
                }
            } else {
                List(selection: $selection) {
                    ForEach(queue.jobs) { (job: QueueJob) in
                        QueueJobRow(queue: queue, job: job)
                            .tag(job.id)
                    }
                    .onMove { (source: IndexSet, destination: Int) in
                        queue.move(fromOffsets: source, toOffset: destination)
                    }
                    .onDelete { (offsets: IndexSet) in
                        queue.remove(Set(offsets.map { queue.jobs[$0].id }))
                    }
                }
            }
        }
        .toolbar {
            ToolbarItemGroup {
                if queue.isRunning {
                    Button {
                        queue.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .help("Stop the queue. The running project keeps its last completed model.")
                } else {
                    Button {
                        queue.runQueue(modelStore: PipelineModel.makeModelStore())
                    } label: {
                        Label("Run Queue", systemImage: "play.fill")
                    }
                    .help("Train the waiting projects one after another")
                    .disabled(!queue.hasWaitingJobs)
                }
                Button {
                    queue.remove(selection)
                    selection = []
                } label: {
                    Label("Remove Selected", systemImage: "minus.circle")
                }
                .help("Remove the selected jobs from the queue")
                .disabled(selection.isEmpty || queue.isRunning)
                Button {
                    queue.clearFinished()
                } label: {
                    Label("Clear Finished", systemImage: "trash")
                }
                .help("Remove the finished and failed jobs")
                .disabled(queue.isRunning)
            }
        }
        .alert("SparseSplat", isPresented: messageIsPresented) {
            Button("OK") {
                queue.message = nil
            }
        } message: {
            Text(queue.message ?? "")
        }
    }

    private var messageIsPresented: Binding<Bool> {
        Binding(get: { queue.message != nil },
                set: { (presented: Bool) in
                    if !presented {
                        queue.message = nil
                    }
                })
    }
}

/// One job: its name, state and, while it runs, its progress. Waiting jobs show their mode and preset as editable pickers.
private struct QueueJobRow: View {
    let queue: TrainingQueue
    let job: QueueJob

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: symbol)
                    .foregroundStyle(tint)
                    .accessibilityHidden(true)
                Text(job.name)
                    .font(.headline)
                Spacer(minLength: 8)
                Text(job.state.title)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if job.state == .running {
                runningProgress
            }
            if job.state == .waiting {
                HStack(spacing: 12) {
                    Picker("Mode", selection: modeBinding) {
                        ForEach(QueueJobMode.allCases, id: \.self) { (mode: QueueJobMode) in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .frame(maxWidth: 260)
                    Picker("Preset", selection: presetBinding) {
                        Text("Stored settings").tag(TrainingPreset?.none)
                        ForEach(TrainingPreset.allCases, id: \.self) { (preset: TrainingPreset) in
                            Text(preset.rawValue.capitalized).tag(TrainingPreset?.some(preset))
                        }
                    }
                    .frame(maxWidth: 240)
                }
            } else {
                Text(job.mode.title + (job.preset.map { ", \($0.rawValue) preset" } ?? ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let message: String = job.message {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var runningProgress: some View {
        if let total: Int = job.totalIterations, total > 0 {
            let done: Int = job.iteration ?? 0
            ProgressView(value: Double(done), total: Double(total))
                .accessibilityLabel("Training progress")
                .accessibilityValue("\(done) of \(total) iterations")
            Text("Iteration \(done) of \(total)")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
        } else {
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel("Preparing training")
        }
    }

    private var modeBinding: Binding<QueueJobMode> {
        Binding(get: { job.mode },
                set: { (mode: QueueJobMode) in queue.setMode(job.id, to: mode) })
    }

    private var presetBinding: Binding<TrainingPreset?> {
        Binding(get: { job.preset },
                set: { (preset: TrainingPreset?) in queue.setPreset(job.id, to: preset) })
    }

    private var symbol: String {
        switch job.state {
        case .waiting: "clock"
        case .running: "gearshape.2"
        case .done: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    private var tint: Color {
        switch job.state {
        case .waiting: .secondary
        case .running: .accentColor
        case .done: .green
        case .failed: .orange
        }
    }
}
