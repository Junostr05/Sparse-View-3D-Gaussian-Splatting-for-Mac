import AppKit
import SplatIO
import SwiftUI

/// One project in the library grid: cover, name, a one-line summary, the modification date and a status badge.
/// The card is content, so it carries no glass.
struct LibraryCard: View {
    let item: LibraryItem
    let isSelected: Bool
    let covers: LibraryCovers

    @State private var cover: NSImage?

    private static let cornerRadius: CGFloat = 14

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            coverView
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(summary)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let date = item.modificationDate {
                    Text(date, style: .relative)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 6)
            .padding(.bottom, 6)
        }
        .padding(6)
        .background {
            RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.16) : Color.clear)
        }
        .overlay {
            RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous)
                .strokeBorder(isSelected ? Color.accentColor : Color.secondary.opacity(0.2),
                              lineWidth: isSelected ? 2 : 1)
        }
        .contentShape(RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous))
        .focusable()
        .task(id: coverKey) {
            cover = await covers.cover(for: item)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.name), \(summary), \(badge.title)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Double-click to open")
    }

    /// Changes when the package is saved, so the cover is read again.
    private var coverKey: String {
        let stamp: Double = item.modificationDate?.timeIntervalSince1970 ?? 0
        return item.url.path(percentEncoded: false) + "|" + String(stamp)
    }

    private var coverView: some View {
        ZStack {
            Rectangle()
                .fill(.quaternary)
            if let cover {
                Image(nsImage: cover)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 40, weight: .regular))
                    .foregroundStyle(.secondary)
            }
        }
        .aspectRatio(4.0 / 3.0, contentMode: .fit)
        .clipped()
        .clipShape(RoundedRectangle(cornerRadius: Self.cornerRadius - 4, style: .continuous))
        .overlay(alignment: .topLeading) {
            statusBadge
                .padding(8)
        }
        .accessibilityHidden(true)
    }

    private var statusBadge: some View {
        Label(badge.title, systemImage: badge.symbol)
            .font(.caption2.weight(.medium))
            .labelStyle(.titleAndIcon)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background {
                Capsule()
                    .fill(Color(nsColor: .controlBackgroundColor).opacity(0.9))
            }
    }

    private var badge: (title: String, symbol: String) {
        if item.isOutsideLibrary {
            return ("Outside Library", "folder")
        }
        guard let metadata = item.metadata else {
            return ("Unreadable", "exclamationmark.triangle")
        }
        if metadata.hasCheckpoint {
            return ("Trained", "checkmark.circle")
        }
        return ("Not Trained", "circle.dashed")
    }

    /// `13 photos · 757 k Gaussians · 31.1 dB`, or `13 photos · Not trained yet`.
    private var summary: String {
        guard let metadata = item.metadata else {
            return "Could not read this project"
        }
        var parts: [String] = []
        if metadata.photoCount > 0 {
            parts.append(metadata.photoCount == 1 ? "1 photo" : "\(metadata.photoCount) photos")
        }
        if metadata.hasCheckpoint, let gaussians = metadata.gaussianCount {
            parts.append(Self.gaussianText(gaussians))
            if let psnr = metadata.psnr {
                parts.append(String(format: "%.1f dB", psnr))
            }
        } else {
            parts.append("Not trained yet")
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    /// `757 k Gaussians` below a million, `1.2 M Gaussians` above.
    static func gaussianText(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1f M Gaussians", Double(count) / 1_000_000)
        }
        if count >= 1_000 {
            return "\(count / 1_000) k Gaussians"
        }
        return "\(count) Gaussians"
    }
}
