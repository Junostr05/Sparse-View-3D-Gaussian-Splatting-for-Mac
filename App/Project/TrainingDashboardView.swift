import Charts
import SwiftUI
import SplatCore

/// Live charts and figures of the current training run: loss on a log scale, PSNR, and the run's numbers.
/// Shown only once the run has recorded a sample, so an untouched project shows nothing. Sized for the inspector.
/// Content only: no glass on the charts or the figures.
struct TrainingDashboardView: View {
    let model: ProjectModel

    var body: some View {
        let history: TrainingHistory = model.history
        if history.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 14) {
                figures(history)
                lossChart(history)
                psnrChart(history)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Figures

    /// Six figures in three rows of two, for the 240 to 380 pt inspector column.
    private func figures(_ history: TrainingHistory) -> some View {
        let gaussians: String = history.latest.map { $0.gaussianCount.formatted() } ?? "\u{2014}"
        let rate: String = history.smoothedIterationsPerSecond > 0
            ? String(format: "%.1f", history.smoothedIterationsPerSecond)
            : "\u{2014}"
        return Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
            GridRow {
                figure("Iteration", value: iterationText)
                figure("Gaussians", value: gaussians)
            }
            GridRow {
                figure("Iterations/s", value: rate)
                figure("Elapsed", value: Self.durationText(history.elapsedSeconds))
            }
            GridRow {
                figure("ETA", value: etaText(history))
                figure("GPU memory", value: Self.megabytesText(model.gpuMemoryBytes))
            }
        }
    }

    private func figure(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// "1,200 of 10,000" for a run with a target, the count alone for an endless continuation.
    private var iterationText: String {
        let done: String = model.totalIterations.formatted()
        guard let target = model.runTotalIterations else {
            return done
        }
        return "\(done) of \(target.formatted())"
    }

    private func etaText(_ history: TrainingHistory) -> String {
        if model.isEndlessRun {
            return "Open-ended"
        }
        guard let seconds = history.estimatedRemainingSeconds(totalIterations: model.runTotalIterations) else {
            return "\u{2014}"
        }
        return Self.durationText(seconds)
    }

    // MARK: - Charts

    private func lossChart(_ history: TrainingHistory) -> some View {
        // Log scale needs positive values; a loss of zero or less is left out of the chart only.
        let points: [ChartPoint] = Self.chartPoints(history) { point in
            guard let loss = point.loss, loss > 0 else {
                return nil
            }
            return loss
        }
        return VStack(alignment: .leading, spacing: 4) {
            Text("Loss")
                .font(.caption.weight(.semibold))
            if points.isEmpty {
                placeholder("Loss appears after the first logged step.")
            } else {
                Chart(points) { point in
                    LineMark(x: .value("Iteration", point.iteration),
                             y: .value("Loss", point.value))
                }
                .chartYScale(type: .log)
                .chartXAxisLabel("Iteration")
                .frame(height: 120)
                .accessibilityLabel(Self.lossSummary(points))
            }
        }
    }

    private func psnrChart(_ history: TrainingHistory) -> some View {
        let points: [ChartPoint] = Self.chartPoints(history) { point in
            point.psnr
        }
        return VStack(alignment: .leading, spacing: 4) {
            Text("PSNR (dB)")
                .font(.caption.weight(.semibold))
            if points.isEmpty {
                placeholder("PSNR appears after the first evaluation.")
            } else {
                Chart(points) { point in
                    LineMark(x: .value("Iteration", point.iteration),
                             y: .value("PSNR", point.value))
                }
                .chartXAxisLabel("Iteration")
                .frame(height: 120)
                .accessibilityLabel(Self.psnrSummary(points))
            }
        }
    }

    private func placeholder(_ text: String) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 60, alignment: .leading)
    }

    // MARK: - Chart data and labels

    /// One plotted value. `id` is the index in the history, which is unique for the points of one run.
    private struct ChartPoint: Identifiable {
        let id: Int
        let iteration: Int
        let value: Double
    }

    /// Points whose selected value exists, in iteration order.
    private static func chartPoints(_ history: TrainingHistory,
                                    _ select: (TrainingHistory.Point) -> Double?) -> [ChartPoint] {
        var result: [ChartPoint] = []
        for (index, point) in history.points.enumerated() {
            guard let value = select(point) else {
                continue
            }
            result.append(ChartPoint(id: index, iteration: point.iteration, value: value))
        }
        return result
    }

    /// "Loss decreasing from 0.17 to 0.03" from the first and the last plotted value.
    private static func lossSummary(_ points: [ChartPoint]) -> String {
        trendSummary(name: "Loss", points: points, format: "%.3g", unit: "")
    }

    /// "PSNR increasing from 18.2 dB to 24.5 dB" from the first and the last plotted value.
    private static func psnrSummary(_ points: [ChartPoint]) -> String {
        trendSummary(name: "PSNR", points: points, format: "%.1f", unit: " dB")
    }

    private static func trendSummary(name: String, points: [ChartPoint], format: String, unit: String) -> String {
        guard let first = points.first, let last = points.last else {
            return "\(name) has no values yet."
        }
        let from: String = String(format: format, first.value) + unit
        let to: String = String(format: format, last.value) + unit
        if last.value < first.value {
            return "\(name) decreasing from \(from) to \(to)"
        }
        if last.value > first.value {
            return "\(name) increasing from \(from) to \(to)"
        }
        return "\(name) unchanged at \(to)"
    }

    /// "1 h 02 min", "4 min 12 s", "38 s". Whole seconds; the first two units only.
    static func durationText(_ seconds: Double) -> String {
        let total: Int = Int(seconds.rounded())
        let hours: Int = total / 3600
        let minutes: Int = (total % 3600) / 60
        let secs: Int = total % 60
        if hours > 0 {
            return String(format: "%ld h %02ld min", hours, minutes)
        }
        if minutes > 0 {
            return "\(minutes) min \(secs) s"
        }
        return "\(secs) s"
    }

    /// "512 MB", or "\u{2014}" when the device reports nothing.
    static func megabytesText(_ bytes: Int) -> String {
        guard bytes > 0 else {
            return "\u{2014}"
        }
        let megabytes: Int = bytes / 1_048_576
        return "\(megabytes.formatted()) MB"
    }
}
