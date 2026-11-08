import AppKit
import SplatIO
import SwiftUI

/// Which projects the library grid shows.
enum LibraryScope: String, CaseIterable, Identifiable {
    case all
    case recent

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .recent: "Recent"
        }
    }
}

/// The library window: the projects of the library folder as a grid of cards, with search, sorting, recents and
/// the actions on the selected projects. Opening a project shows its own document window; this window stays open.
struct LibraryView: View {
    let folder: LibraryFolder
    let store: LibraryStore
    let covers: LibraryCovers

    @State private var scope: LibraryScope = .all
    @State private var searchText: String = ""
    @AppStorage("librarySortOrder") private var sortOrder: LibrarySortOrder = .dateModified
    @State private var selection: Set<URL> = []
    /// The last plain click, the start of a Shift-click range.
    @State private var anchor: URL?
    @State private var isNewProjectPresented: Bool = false
    @State private var renameTarget: LibraryItem?
    @State private var isDropTargeted: Bool = false

    var body: some View {
        content
            .navigationTitle("Library")
            .toolbar {
                toolbarContent
            }
            .sheet(isPresented: $isNewProjectPresented) {
                if let url = folder.readyURL {
                    NewProjectSheet(folder: url,
                                    store: store,
                                    existingNames: store.items.map(\.name))
                }
            }
            .sheet(item: $renameTarget) { (item: LibraryItem) in
                if let url = folder.readyURL {
                    RenameProjectSheet(item: item,
                                       takenNames: item.isOutsideLibrary ? [] : store.items.map(\.name),
                                       onRename: { (name: String) async -> Bool in
                                           await store.rename(item.url, to: name, folder: url) != nil
                                       })
                }
            }
            .alert("SparseSplat", isPresented: messageIsPresented) {
                Button("OK") {
                    store.message = nil
                }
            } message: {
                Text(store.message ?? "")
            }
            .task(id: folder.readyURL) {
                await refreshFromFolder()
            }
            .task {
                // Refresh when the app becomes active again, for example after a project was edited in Finder.
                for await _ in NotificationCenter.default.notifications(named: NSApplication.didBecomeActiveNotification) {
                    await refreshFromFolder()
                }
            }
            .onChange(of: store.isNewProjectRequested, initial: true) { (_, requested: Bool) in
                if requested {
                    store.isNewProjectRequested = false
                    isNewProjectPresented = true
                }
            }
            .dropDestination(for: URL.self) { (urls: [URL], _: CGPoint) -> Bool in
                handleDrop(urls)
            } isTargeted: { (targeted: Bool) in
                isDropTargeted = targeted
            }
            .overlay {
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Color.accentColor, lineWidth: 3)
                        .padding(8)
                        .allowsHitTesting(false)
                }
            }
    }

    private var messageIsPresented: Binding<Bool> {
        Binding(get: { store.message != nil },
                set: { (presented: Bool) in
                    if !presented {
                        store.message = nil
                    }
                })
    }

    @ViewBuilder
    private var content: some View {
        switch folder.state {
        case .unset:
            LibraryOnboardingView(folder: folder)
        case .missing(let url):
            LibraryFolderProblemView(
                title: "Library Folder Not Found",
                message: "\(url.path(percentEncoded: false)) can no longer be found. It may have been moved, renamed or disconnected.",
                folder: folder)
        case .unresolvable:
            LibraryFolderProblemView(
                title: "Library Folder Unavailable",
                message: "The library folder could not be opened. Choose it again.",
                folder: folder)
        case .ready:
            projectGrid
                .searchable(text: $searchText, placement: .toolbar, prompt: "Search projects")
        }
    }

    // MARK: - Grid

    /// The projects the grid shows: the folder's projects sorted by the sort menu, or the recent projects by date.
    private var visibleItems: [LibraryItem] {
        switch scope {
        case .all:
            let matching: [LibraryItem] = store.items.filter { LibrarySearch.matches($0.name, query: searchText) }
            return LibrarySortOrder.sorted(matching, by: sortOrder)
        case .recent:
            return store.recentItems.filter { LibrarySearch.matches($0.name, query: searchText) }
        }
    }

    @ViewBuilder
    private var projectGrid: some View {
        let shown: [LibraryItem] = visibleItems
        if shown.isEmpty {
            emptyState
        } else {
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220, maximum: 320), spacing: 16)], spacing: 16) {
                    ForEach(shown) { (item: LibraryItem) in
                        LibraryCard(item: item,
                                    isSelected: selection.contains(item.url),
                                    covers: covers)
                            .onTapGesture(count: 2) {
                                select(item, modifiers: [])
                                open(targets(for: item))
                            }
                            .onTapGesture {
                                select(item, modifiers: NSEvent.modifierFlags)
                            }
                            .contextMenu {
                                cardMenu(for: item)
                            }
                    }
                }
                .padding(16)
            }
            .onKeyPress(.return) {
                openSelection()
                return .handled
            }
            .onDeleteCommand {
                trash(selectedURLs())
            }
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        if !searchText.isEmpty {
            ContentUnavailableView.search(text: searchText)
        } else if scope == .recent {
            ContentUnavailableView("No Recent Projects",
                                   systemImage: "clock",
                                   description: Text("Projects you open appear here."))
        } else {
            ContentUnavailableView {
                Label("No Projects Yet", systemImage: "square.stack.3d.up")
            } description: {
                Text("Create a project, or drop a .gsproj package onto this window.")
            } actions: {
                Button("New Project\u{2026}") {
                    isNewProjectPresented = true
                }
                .buttonStyle(.glassProminent)
            }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        if folder.readyURL != nil {
            ToolbarItem(placement: .navigation) {
                Picker("Show", selection: $scope) {
                    ForEach(LibraryScope.allCases) { (option: LibraryScope) in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .accessibilityLabel("Show all projects or recent projects")
            }
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Picker("Sort By", selection: $sortOrder) {
                        Text("Name").tag(LibrarySortOrder.name)
                        Text("Date Modified").tag(LibrarySortOrder.dateModified)
                        Text("Last Opened").tag(LibrarySortOrder.lastOpened)
                        Text("Size").tag(LibrarySortOrder.size)
                    }
                    .pickerStyle(.inline)
                } label: {
                    Label("Sort", systemImage: "arrow.up.arrow.down")
                }
                .help("Sort projects")
                .disabled(scope == .recent)
            }
            if scope == .recent {
                ToolbarItem(placement: .primaryAction) {
                    Button("Clear Recents") {
                        guard let url = folder.readyURL else {
                            return
                        }
                        Task {
                            await store.clearRecents(folder: url)
                        }
                    }
                    .help("Remove all projects from Recent")
                }
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    open(selectedURLs())
                } label: {
                    Label("Open", systemImage: "doc")
                }
                .help("Open the selected projects")
                .disabled(selection.isEmpty)
                Button {
                    if let item = selectedItems().first {
                        renameTarget = item
                    }
                } label: {
                    Label("Rename", systemImage: "pencil")
                }
                .help("Rename the selected project")
                .disabled(selection.count != 1)
                Button {
                    duplicate(selectedURLs())
                } label: {
                    Label("Duplicate", systemImage: "plus.square.on.square")
                }
                .help("Copy the selected projects")
                .disabled(selection.isEmpty)
                Button {
                    store.reveal(selectedURLs())
                } label: {
                    Label("Reveal in Finder", systemImage: "folder")
                }
                .help("Show the selected projects in the Finder")
                .disabled(selection.isEmpty)
                Button(role: .destructive) {
                    trash(selectedURLs())
                } label: {
                    Label("Move to Trash", systemImage: "trash")
                }
                .help("Move the selected projects to the Trash")
                .disabled(selection.isEmpty)
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        await refreshFromFolder()
                    }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .help("Reload the projects of the library")
                .keyboardShortcut("r", modifiers: .command)
            }
            ToolbarItem(placement: .primaryAction) {
                Button("New Project\u{2026}") {
                    isNewProjectPresented = true
                }
                .buttonStyle(.glassProminent)
                .help("Create an empty project in the library")
            }
        }
    }

    // MARK: - Menus

    @ViewBuilder
    private func cardMenu(for item: LibraryItem) -> some View {
        let urls: [URL] = targets(for: item)
        Button("Open") {
            open(urls)
        }
        if urls.count == 1 {
            Button("Rename\u{2026}") {
                renameTarget = item
            }
        }
        Button("Duplicate") {
            duplicate(urls)
        }
        Button("Reveal in Finder") {
            store.reveal(urls)
        }
        Divider()
        Button("Move to Trash", role: .destructive) {
            trash(urls)
        }
    }

    // MARK: - Selection

    /// The items of the grid that are selected, in grid order.
    private func selectedItems() -> [LibraryItem] {
        visibleItems.filter { selection.contains($0.url) }
    }

    private func selectedURLs() -> [URL] {
        selectedItems().map(\.url)
    }

    /// The projects a menu acts on: the whole selection when `item` is part of it, otherwise `item` alone.
    private func targets(for item: LibraryItem) -> [URL] {
        if selection.contains(item.url) {
            return selectedURLs()
        }
        return [item.url]
    }

    /// Selects `item`. Command toggles it in the selection, Shift selects the range from the anchor, and a plain
    /// click selects only `item`.
    private func select(_ item: LibraryItem, modifiers: NSEvent.ModifierFlags) {
        if modifiers.contains(.command) {
            if selection.contains(item.url) {
                selection.remove(item.url)
            } else {
                selection.insert(item.url)
            }
            anchor = item.url
        } else if modifiers.contains(.shift), let anchor {
            let urls: [URL] = visibleItems.map(\.url)
            guard let from = urls.firstIndex(of: anchor), let to = urls.firstIndex(of: item.url) else {
                selection = [item.url]
                return
            }
            selection = Set(urls[min(from, to)...max(from, to)])
        } else {
            selection = [item.url]
            anchor = item.url
        }
    }

    // MARK: - Actions

    private func openSelection() {
        open(selectedURLs())
    }

    private func open(_ urls: [URL]) {
        guard let url = folder.readyURL, !urls.isEmpty else {
            return
        }
        store.open(urls, folder: url)
    }

    private func duplicate(_ urls: [URL]) {
        guard let url = folder.readyURL, !urls.isEmpty else {
            return
        }
        Task {
            await store.duplicate(urls, folder: url)
        }
    }

    private func trash(_ urls: [URL]) {
        guard let url = folder.readyURL, !urls.isEmpty else {
            return
        }
        selection.subtract(urls)
        Task {
            await store.moveToTrash(urls, folder: url)
        }
    }

    /// Copies dropped packages into the library folder. Returns `false` when the drop holds no package.
    private func handleDrop(_ urls: [URL]) -> Bool {
        guard let url = folder.readyURL else {
            return false
        }
        let packages: [URL] = urls.filter { $0.pathExtension.lowercased() == LibraryNaming.packageExtension }
        guard !packages.isEmpty else {
            return false
        }
        Task {
            _ = await store.importDropped(packages, folder: url)
        }
        return true
    }

    private func refreshFromFolder() async {
        guard let url = folder.readyURL else {
            return
        }
        await store.refresh(folder: url)
    }
}
