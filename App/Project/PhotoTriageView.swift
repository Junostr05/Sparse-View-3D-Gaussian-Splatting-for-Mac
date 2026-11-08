import AppKit
import SwiftUI
import SplatCore
import SplatIO

/// Runs photo triage for one project: decodes each photo at a reduced size, measures it, asks `PhotoTriage` for
/// verdicts and keeps a small thumbnail per photo. Analysis runs photo by photo, off the main actor for the decode.
@Observable
final class PhotoTriageModel {
    private(set) var verdicts: [String: PhotoVerdict] = [:]
    private(set) var thumbnails: [String: CGImage] = [:]
    private(set) var isAnalyzing: Bool = false
    private(set) var analyzedCount: Int = 0
    private(set) var totalCount: Int = 0

    /// The verdict of one photo, or `nil` while it is unanalysed or could not be read.
    func verdict(for name: String) -> PhotoVerdict? {
        verdicts[name]
    }

    /// Measures every photo in `contents` and assigns verdicts. Cancelling the task stops after the current photo.
    func analyze(contents: ProjectContents) async {
        let names: [String] = contents.manifest.imageFiles
        isAnalyzing = true
        analyzedCount = 0
        totalCount = names.count
        verdicts = [:]
        thumbnails = [:]

        var inputs: [PhotoInput] = []
        for name in names {
            if Task.isCancelled {
                break
            }
            if let data = contents.files[ProjectLayout.imagesDirectory + "/" + name] {
                let metrics: PhotoMetrics? = await Task.detached(priority: .userInitiated) { () -> PhotoMetrics? in
                    guard let image = try? ImageIODecoder.decodeThumbnail(data, maxSide: PhotoQuality.analysisSide) else {
                        return nil
                    }
                    return PhotoQuality.measure(image)
                }.value
                if let metrics {
                    inputs.append(PhotoInput(name: name, metrics: metrics))
                }
                if let thumbnail = ImageIODecoder.thumbnail(data, maxSide: 96) {
                    thumbnails[name] = thumbnail
                }
            }
            analyzedCount += 1
        }

        if !Task.isCancelled {
            let poses: [NamedCamera] = (try? ProjectBundle.cameraPoses(contents)) ?? []
            var cameras: [String: Camera] = [:]
            for pose in poses where cameras[pose.name] == nil {
                cameras[pose.name] = pose.camera
            }
            let results: [PhotoVerdict] = PhotoTriage.assess(images: inputs, cameras: cameras)
            var byName: [String: PhotoVerdict] = [:]
            for result in results {
                byName[result.name] = result
            }
            verdicts = byName
        }
        isAnalyzing = false
    }
}

/// The "Photo Triage" section of the Capture tab: one row per photo with its thumbnail, verdict and a Use toggle.
/// The verdicts are computed once per set of photos and do not change when the user toggles photos.
struct PhotoTriageSection: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    @State private var triage = PhotoTriageModel()

    var body: some View {
        let contents: ProjectContents = document.wrappedValue.contents
        let names: [String] = contents.manifest.imageFiles
        let excluded: Set<String> = Set(contents.manifest.excludedImages)
        let usedCount: Int = names.count - names.filter { excluded.contains($0) }.count
        let hasCheckpoint: Bool = !contents.manifest.checkpoints.isEmpty

        Section {
            if triage.isAnalyzing {
                HStack(spacing: 8) {
                    ProgressView(value: Double(triage.analyzedCount), total: Double(max(triage.totalCount, 1)))
                        .accessibilityLabel("Checking photos")
                    Text("\(triage.analyzedCount) of \(triage.totalCount)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            Text(summaryText(names: names, usedCount: usedCount))
                .font(.footnote)
                .foregroundStyle(.secondary)

            ForEach(names, id: \.self) { name in
                photoRow(name: name, isUsed: !excluded.contains(name), usedCount: usedCount)
            }

            Button("Auto-exclude Flagged") {
                autoExclude(names: names)
            }
            .buttonStyle(.glass)
            .disabled(model.isTraining || triage.isAnalyzing || flaggedNames(names).isEmpty)
            .help("Excludes every photo that is blurry, badly exposed or a duplicate. The photos stay in the project.")

            if hasCheckpoint && !excluded.isEmpty {
                Text("Retrain to apply. The saved model was trained with the photos it used before.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Photo Triage")
        }
        .task(id: names) {
            await triage.analyze(contents: contents)
        }
    }

    // MARK: - Rows

    private func photoRow(name: String, isUsed: Bool, usedCount: Int) -> some View {
        let verdict: PhotoVerdict? = triage.verdict(for: name)
        let isLastUsed: Bool = isUsed && usedCount <= 1
        return HStack(alignment: .center, spacing: 10) {
            thumbnail(for: name)
            VStack(alignment: .leading, spacing: 2) {
                Text(name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                verdictLabel(verdict)
            }
            Spacer(minLength: 4)
            Toggle("Use", isOn: useBinding(name))
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(model.isTraining || isLastUsed)
                .accessibilityLabel("Use \(name) for training")
                .help(isLastUsed ? "At least one photo must be used." : "Turn off to leave this photo out of training.")
        }
    }

    @ViewBuilder
    private func verdictLabel(_ verdict: PhotoVerdict?) -> some View {
        if let verdict {
            VStack(alignment: .leading, spacing: 1) {
                Label(Self.badgeText(verdict.kind), systemImage: Self.badgeSymbol(verdict.kind))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(verdict.isOK ? Color.secondary : Color.orange)
                Text(verdict.reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } else if triage.isAnalyzing {
            Text("Checking\u{2026}")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Text("Not checked")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func thumbnail(for name: String) -> some View {
        if let image = triage.thumbnails[name] {
            Image(nsImage: NSImage(cgImage: image, size: .zero))
                .resizable()
                .scaledToFill()
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .accessibilityHidden(true)
        } else {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.secondary.opacity(0.2))
                .frame(width: 44, height: 44)
                .accessibilityHidden(true)
        }
    }

    // MARK: - Summary and actions

    private func summaryText(names: [String], usedCount: Int) -> String {
        let total: Int = names.count
        var parts: [String] = ["\(usedCount) of \(total) photos used"]
        if !triage.isAnalyzing && !triage.verdicts.isEmpty {
            let flagged: [PhotoVerdict] = names.compactMap { triage.verdict(for: $0) }.filter { !$0.isOK }
            if flagged.isEmpty {
                parts.append("no problems found")
            } else {
                parts.append("\(flagged.count) flagged")
                let blurry: Int = flagged.filter { $0.kind == .blurry }.count
                let exposure: Int = flagged.filter { $0.kind == .tooDark || $0.kind == .tooBright }.count
                let duplicates: Int = flagged.filter { if case .duplicate = $0.kind { return true } else { return false } }.count
                var details: [String] = []
                if blurry > 0 { details.append("\(blurry) blurry") }
                if exposure > 0 { details.append("\(exposure) exposure") }
                if duplicates > 0 { details.append("\(duplicates) duplicate") }
                parts.append("(" + details.joined(separator: ", ") + ")")
            }
        }
        return parts.joined(separator: ", ")
    }

    private func flaggedNames(_ names: [String]) -> [String] {
        names.filter { name in
            if let verdict = triage.verdict(for: name) {
                return !verdict.isOK
            }
            return false
        }
    }

    /// Excludes every flagged photo. Leaves the project alone when that would exclude every photo.
    private func autoExclude(names: [String]) {
        let flagged: [String] = flaggedNames(names)
        guard !flagged.isEmpty, flagged.count < names.count else {
            return
        }
        document.wrappedValue.contents.manifest.excludedImages = flagged.sorted()
    }

    private func useBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { !document.wrappedValue.contents.manifest.excludedImages.contains(name) },
            set: { used in
                var excluded: [String] = document.wrappedValue.contents.manifest.excludedImages
                if used {
                    excluded.removeAll { $0 == name }
                } else if !excluded.contains(name) {
                    excluded.append(name)
                }
                document.wrappedValue.contents.manifest.excludedImages = excluded.sorted()
            }
        )
    }

    // MARK: - Verdict presentation

    private static func badgeText(_ kind: PhotoVerdictKind) -> String {
        switch kind {
        case .ok: "Good"
        case .blurry: "Blurry"
        case .tooDark: "Too dark"
        case .tooBright: "Too bright"
        case .duplicate: "Duplicate"
        }
    }

    private static func badgeSymbol(_ kind: PhotoVerdictKind) -> String {
        switch kind {
        case .ok: "checkmark.circle"
        case .blurry: "eye.slash"
        case .tooDark: "moon.fill"
        case .tooBright: "sun.max.fill"
        case .duplicate: "square.on.square"
        }
    }
}
