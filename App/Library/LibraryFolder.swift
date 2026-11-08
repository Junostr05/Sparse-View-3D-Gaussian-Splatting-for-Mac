import AppKit
import Foundation
import Observation

/// The library folder the user chose. Stored as an app-scoped security-scoped bookmark in UserDefaults, so it
/// survives launches in the sandbox. The folder's access is started once and kept until the folder changes.
@Observable
final class LibraryFolder {
    enum State: Equatable {
        /// No folder chosen yet: the library shows onboarding.
        case unset
        /// The folder exists and is accessible.
        case ready(URL)
        /// The bookmark resolves, but the folder is gone (deleted, renamed or on an unmounted volume).
        case missing(URL)
        /// The bookmark does not resolve at all. The user has to choose the folder again.
        case unresolvable
    }

    /// Name of the folder created by "Use Default Location", inside the app container's Documents directory.
    static let defaultFolderName: String = "SparseSplat Library"

    private static let bookmarkKey: String = "libraryFolderBookmark"

    private(set) var state: State = .unset
    /// The folder whose security-scoped access is running. Stopped when another folder is chosen.
    @ObservationIgnored private var scopedFolder: URL?

    init() {
        restore()
    }

    /// The folder when it is ready, `nil` otherwise.
    var readyURL: URL? {
        if case .ready(let url) = state {
            return url
        }
        return nil
    }

    /// The path shown in the UI, when a folder is known.
    var displayPath: String? {
        switch state {
        case .unset, .unresolvable:
            return nil
        case .ready(let url), .missing(let url):
            return url.path(percentEncoded: false)
        }
    }

    /// Resolves the stored bookmark at launch. A stale bookmark is renewed when its folder is still there.
    func restore() {
        guard let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) else {
            state = .unset
            return
        }
        var isStale: Bool = false
        guard let url = try? URL(resolvingBookmarkData: data,
                                 options: [.withSecurityScope],
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &isStale) else {
            stopScope()
            state = .unresolvable
            return
        }
        adopt(url, renewBookmark: isStale)
    }

    /// Makes `url` the library folder and stores its bookmark.
    func choose(_ url: URL) {
        guard let data = try? url.bookmarkData(options: [.withSecurityScope],
                                               includingResourceValuesForKeys: nil,
                                               relativeTo: nil) else {
            stopScope()
            state = .unresolvable
            return
        }
        UserDefaults.standard.set(data, forKey: Self.bookmarkKey)
        adopt(url, renewBookmark: false)
    }

    /// Shows the open panel for a folder. Folders can be created in the panel.
    func chooseWithPanel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose the folder where your SparseSplat projects are kept."
        if panel.runModal() == .OK, let url = panel.url {
            choose(url)
        }
    }

    /// Creates `SparseSplat Library` in the app container's Documents directory (if needed) and uses it.
    func useDefaultLocation() {
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            state = .unresolvable
            return
        }
        let folder: URL = documents.appending(path: Self.defaultFolderName, directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        } catch {
            state = .unresolvable
            return
        }
        choose(folder)
    }

    /// Starts the folder's security-scoped access, checks that it exists and sets the state.
    private func adopt(_ url: URL, renewBookmark: Bool) {
        stopScope()
        if url.startAccessingSecurityScopedResource() {
            scopedFolder = url
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false), isDirectory: &isDirectory),
              isDirectory.boolValue else {
            state = .missing(url)
            return
        }
        if renewBookmark,
           let data = try? url.bookmarkData(options: [.withSecurityScope],
                                            includingResourceValuesForKeys: nil,
                                            relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: Self.bookmarkKey)
        }
        state = .ready(url)
    }

    private func stopScope() {
        scopedFolder?.stopAccessingSecurityScopedResource()
        scopedFolder = nil
    }
}
