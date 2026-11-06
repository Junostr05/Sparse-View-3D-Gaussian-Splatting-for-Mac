import SwiftUI

/// The frame strip: the candidate frames of a photo folder or a video as a thumbnail grid. Selected frames
/// have an accent border and a check mark. This is content, so it carries no glass.
struct FrameStripView: View {
    let pipeline: PipelineModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Click a frame to include it or leave it out.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)
                .padding(.top, 12)
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 12, alignment: .top)],
                          alignment: .leading,
                          spacing: 12) {
                    ForEach(pipeline.candidates) { candidate in
                        FrameCell(pipeline: pipeline, candidate: candidate)
                    }
                }
                .padding(16)
            }
        }
    }
}

/// One candidate frame: thumbnail, caption and a toggle button for its selection.
private struct FrameCell: View {
    let pipeline: PipelineModel
    let candidate: PipelineCandidate

    var body: some View {
        let selected: Bool = pipeline.isSelected(candidate.id)
        Button {
            pipeline.toggle(candidate.id)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                thumbnail(selected: selected)
                Text(candidate.caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(6)
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(selected ? Color.accentColor : Color.clear, lineWidth: 3)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(PipelineModel.frameLabel(candidate, selected: selected))
        .accessibilityAddTraits(.isButton)
        .task(id: candidate.id) {
            await pipeline.loadThumbnail(id: candidate.id)
        }
    }

    @ViewBuilder
    private func thumbnail(selected: Bool) -> some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if let image = pipeline.thumbnails[candidate.id] {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFill()
                } else {
                    Rectangle()
                        .fill(.quaternary)
                        .overlay {
                            ProgressView()
                                .controlSize(.small)
                        }
                }
            }
            .aspectRatio(4.0 / 3.0, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            if selected {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.white, Color.accentColor)
                    .padding(6)
                    .accessibilityHidden(true)
            }
        }
    }
}
