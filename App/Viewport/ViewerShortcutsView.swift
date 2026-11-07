import SwiftUI

/// A short reference of the viewport controls (Blender mapping). Shown in a popover.
struct ViewerShortcutsView: View {
    private struct Entry: Identifiable {
        let id = UUID()
        let keys: String
        let action: String
    }

    private struct Group: Identifiable {
        let id = UUID()
        let title: String
        let entries: [Entry]
    }

    private static let groups: [Group] = [
        Group(title: "Mouse", entries: [
            Entry(keys: "Middle drag", action: "Orbit"),
            Entry(keys: "\u{21E7} + middle drag", action: "Pan"),
            Entry(keys: "\u{2303} + middle drag", action: "Zoom"),
            Entry(keys: "Wheel", action: "Zoom"),
            Entry(keys: "\u{2303} + wheel / \u{21E7} + wheel", action: "Pan horizontally / vertically"),
            Entry(keys: "Left drag", action: "Orbit (\u{2325} pans)"),
            Entry(keys: "Right drag", action: "Pan"),
            Entry(keys: "Press or first scroll", action: "Pivot at the surface under the cursor"),
        ]),
        Group(title: "Trackpad", entries: [
            Entry(keys: "Two-finger scroll", action: "Orbit"),
            Entry(keys: "\u{21E7} + scroll", action: "Pan"),
            Entry(keys: "\u{2303} or \u{2318} + scroll, pinch", action: "Zoom"),
        ]),
        Group(title: "Views", entries: [
            Entry(keys: "1 / 3 / 7", action: "Front / Right / Top"),
            Entry(keys: "\u{2303} + 1 / 3 / 7", action: "Back / Left / Bottom"),
            Entry(keys: "4 / 6 / 8 / 2", action: "Orbit in 15\u{00B0} steps"),
            Entry(keys: "\u{2303} + 4 / 6 / 8 / 2", action: "Pan in steps"),
            Entry(keys: ". or Home", action: "Frame the scene"),
        ]),
        Group(title: "Editing (Edit Mode)", entries: [
            Entry(keys: "Left drag", action: "Paint the selection (\u{21E7} subtracts)"),
            Entry(keys: "Delete or X", action: "Delete the selection"),
            Entry(keys: "A / \u{2325} + A / \u{2303} + I", action: "Select all / clear / invert"),
            Entry(keys: "F", action: "Front-only brush on / off"),
            Entry(keys: "\u{2318} + Z / \u{21E7}\u{2318} + Z", action: "Undo / redo"),
        ]),
        Group(title: "Fly Mode", entries: [
            Entry(keys: "\u{21E7} + `", action: "Fly mode on / off"),
            Entry(keys: "W / A / S / D", action: "Forward / left / back / right"),
            Entry(keys: "Q / E", action: "Down / up"),
            Entry(keys: "Mouse", action: "Look around"),
            Entry(keys: "\u{21E7} / \u{2325}", action: "4\u{00D7} faster / 4\u{00D7} slower"),
            Entry(keys: "Wheel", action: "Fly speed"),
            Entry(keys: "Esc, Return or click", action: "Leave fly mode"),
        ]),
        Group(title: "Cameras", entries: [
            Entry(keys: "0", action: "View from the current camera"),
            Entry(keys: "[ / ]", action: "Previous / next camera"),
            Entry(keys: "Space", action: "Play or pause the flight"),
        ]),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Viewport Shortcuts")
                .font(.headline)
            ForEach(Self.groups) { group in
                VStack(alignment: .leading, spacing: 4) {
                    Text(group.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(group.entries) { entry in
                        HStack(alignment: .firstTextBaseline) {
                            Text(entry.keys)
                                .font(.system(.body, design: .monospaced))
                                .frame(width: 190, alignment: .leading)
                            Text(entry.action)
                        }
                    }
                }
            }
            Text("Digits work on the top row and on the numpad.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .fixedSize()
    }
}

/// A button that shows `ViewerShortcutsView` in a popover.
struct ViewerShortcutsButton: View {
    @State private var isPresented = false

    var body: some View {
        Button {
            isPresented.toggle()
        } label: {
            Label("Viewport Shortcuts", systemImage: "keyboard")
        }
        .popover(isPresented: $isPresented) {
            ViewerShortcutsView()
        }
    }
}
