import SplatIO
import SwiftUI

/// Shown on first run, before any library folder is chosen.
struct LibraryOnboardingView: View {
    let folder: LibraryFolder

    var body: some View {
        ContentUnavailableView {
            Label("Choose a Library Folder", systemImage: "square.stack.3d.up")
        } description: {
            Text("SparseSplat keeps all your projects in one folder. Choose one to get started.")
        } actions: {
            Button("Choose Library Folder\u{2026}") {
                folder.chooseWithPanel()
            }
            .buttonStyle(.glassProminent)
            .keyboardShortcut(.defaultAction)
            Button("Use Default Location") {
                folder.useDefaultLocation()
            }
            .buttonStyle(.glass)
        }
    }
}

/// Shown when the saved library folder is missing or its bookmark no longer resolves.
struct LibraryFolderProblemView: View {
    let title: String
    let message: String
    let folder: LibraryFolder

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: "folder.badge.questionmark")
        } description: {
            Text(message)
        } actions: {
            Button("Choose Folder\u{2026}") {
                folder.chooseWithPanel()
            }
            .buttonStyle(.glassProminent)
            .keyboardShortcut(.defaultAction)
            Button("Use Default Location") {
                folder.useDefaultLocation()
            }
            .buttonStyle(.glass)
        }
    }
}

/// Name and Create of a new project. The project is written into the library folder and opened.
struct NewProjectSheet: View {
    let folder: URL
    let store: LibraryStore
    let existingNames: [String]

    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var isCreating: Bool = false

    init(folder: URL, store: LibraryStore, existingNames: [String]) {
        self.folder = folder
        self.store = store
        self.existingNames = existingNames
        _name = State(initialValue: LibraryNaming.uniqueName(base: LibraryNaming.defaultProjectName,
                                                             existing: existingNames))
    }

    private var trimmedName: String {
        LibraryNaming.trimmed(name)
    }

    /// Why the name cannot be used, or `nil` when it can.
    private var problem: String? {
        if !LibraryNaming.isValidName(trimmedName) {
            return "Use a name without slashes or colons, up to \(LibraryNaming.maxNameLength) characters."
        }
        if existingNames.contains(where: { $0.lowercased() == trimmedName.lowercased() }) {
            return "A project with this name already exists."
        }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("New Project")
                .font(.headline)
            TextField("Name", text: $name)
                .onSubmit {
                    create()
                }
                .accessibilityLabel("Project name")
            if let problem, !trimmedName.isEmpty {
                Text(problem)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text("The project is created in your library. Add photos in its window, then train it.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Create") {
                    create()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(problem != nil || isCreating)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private func create() {
        guard problem == nil, !isCreating else {
            return
        }
        isCreating = true
        let chosen: String = trimmedName
        Task {
            let created: Bool = await store.createProject(named: chosen, folder: folder)
            isCreating = false
            if created {
                dismiss()
            }
        }
    }
}

/// Renames a project package in its folder. `takenNames` are the names the new name must not equal.
struct RenameProjectSheet: View {
    let item: LibraryItem
    let takenNames: [String]
    let onRename: (String) async -> Bool

    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var isRenaming: Bool = false

    init(item: LibraryItem, takenNames: [String], onRename: @escaping (String) async -> Bool) {
        self.item = item
        self.takenNames = takenNames
        self.onRename = onRename
        _name = State(initialValue: item.name)
    }

    private var trimmedName: String {
        LibraryNaming.trimmed(name)
    }

    private var problem: String? {
        if !LibraryNaming.isValidName(trimmedName) {
            return "Use a name without slashes or colons, up to \(LibraryNaming.maxNameLength) characters."
        }
        if trimmedName != item.name, takenNames.contains(where: { $0.lowercased() == trimmedName.lowercased() }) {
            return "A project with this name already exists."
        }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Rename Project")
                .font(.headline)
            TextField("Name", text: $name)
                .onSubmit {
                    rename()
                }
                .accessibilityLabel("New project name")
            if let problem, !trimmedName.isEmpty {
                Text(problem)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Rename") {
                    rename()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(problem != nil || isRenaming)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private func rename() {
        guard problem == nil, !isRenaming else {
            return
        }
        if trimmedName == item.name {
            dismiss()
            return
        }
        isRenaming = true
        let chosen: String = trimmedName
        Task {
            let renamed: Bool = await onRename(chosen)
            isRenaming = false
            if renamed {
                dismiss()
            }
        }
    }
}
