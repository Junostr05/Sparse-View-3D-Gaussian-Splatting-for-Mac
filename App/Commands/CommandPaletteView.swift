import SplatCore
import SwiftUI

/// What the View menu's Command Palette item asks the focused window to do.
struct ShowCommandPaletteAction {
    let perform: @MainActor () -> Void
}

private struct ShowCommandPaletteKey: FocusedValueKey {
    typealias Value = ShowCommandPaletteAction
}

extension FocusedValues {
    /// Set by a window that offers the command palette. Nil when no such window has focus.
    var showCommandPalette: ShowCommandPaletteAction? {
        get { self[ShowCommandPaletteKey.self] }
        set { self[ShowCommandPaletteKey.self] = newValue }
    }
}

/// Presents the command palette as a sheet over a window and publishes its show action as a focused scene value,
/// so that the menu command acts on the front window.
struct CommandPaletteModifier: ViewModifier {
    /// Builds the commands from the live state of the window. The palette calls it on every evaluation.
    let makeCommands: (OpenWindowAction) -> [AppCommand]

    @Environment(\.openWindow) private var openWindow
    @State private var isPresented = false
    /// The command chosen in the palette. It runs once the sheet has closed, so a sheet that the command
    /// presents does not collide with the palette.
    @State private var pending: AppCommand?

    func body(content: Content) -> some View {
        content
            .focusedSceneValue(\.showCommandPalette, ShowCommandPaletteAction(perform: { isPresented = true }))
            .sheet(isPresented: $isPresented, onDismiss: runPending) {
                CommandPaletteView(commands: { makeCommands(openWindow) }) { command in
                    pending = command
                    isPresented = false
                }
            }
    }

    private func runPending() {
        let command: AppCommand? = pending
        pending = nil
        command?.action()
    }
}

extension View {
    /// Adds the command palette to a window. `makeCommands` builds the commands from the window's live models.
    func commandPalette(_ makeCommands: @escaping (OpenWindowAction) -> [AppCommand]) -> some View {
        modifier(CommandPaletteModifier(makeCommands: makeCommands))
    }
}

/// The command palette: a search field and the matching commands. Up and down move the selection, Return runs
/// it, Escape closes the palette and a click runs the clicked command.
struct CommandPaletteView: View {
    /// Builds the commands on every evaluation, so that the enabled states follow the live models.
    let commands: () -> [AppCommand]
    /// Called with the chosen command. The presenter closes the palette and runs the command.
    let onRun: (AppCommand) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var query: String = ""
    @State private var selection: Int = 0
    @FocusState private var isSearchFocused: Bool

    var body: some View {
        let results: [AppCommand] = CommandSearch.rank(query: query, commands: commands())
        let highlighted: Int = min(selection, max(results.count - 1, 0))
        VStack(alignment: .leading, spacing: 0) {
            TextField("Type a command", text: $query)
                .textFieldStyle(.plain)
                .font(.title3)
                .focused($isSearchFocused)
                .padding(14)
                .onSubmit {
                    if results.indices.contains(highlighted) {
                        onRun(results[highlighted])
                    }
                }
                .onKeyPress(.upArrow) {
                    selection = max(highlighted - 1, 0)
                    return .handled
                }
                .onKeyPress(.downArrow) {
                    selection = min(highlighted + 1, max(results.count - 1, 0))
                    return .handled
                }
                .onKeyPress(.escape) {
                    dismiss()
                    return .handled
                }
                .onChange(of: query) {
                    selection = 0
                }
                .accessibilityLabel("Command search")

            Divider()

            if results.isEmpty {
                ContentUnavailableView.search(text: query)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(results.enumerated()), id: \.element.id) { index, command in
                                row(command, isHighlighted: index == highlighted)
                                    .id(index)
                            }
                        }
                        .padding(6)
                    }
                    .onChange(of: highlighted) {
                        proxy.scrollTo(highlighted)
                    }
                }
            }
        }
        .frame(width: 560, height: 420)
        .onAppear {
            isSearchFocused = true
        }
    }

    private func row(_ command: AppCommand, isHighlighted: Bool) -> some View {
        Button {
            onRun(command)
        } label: {
            HStack(spacing: 10) {
                Text(command.category)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 70, alignment: .leading)
                Text(command.title)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Spacer(minLength: 12)
                if let shortcut = command.shortcut {
                    Text(shortcut)
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(
                isHighlighted ? Color.accentColor.opacity(0.2) : Color.clear,
                in: RoundedRectangle(cornerRadius: 6, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityHint(command.category)
    }
}
