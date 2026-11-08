import AppKit
import Foundation
import Observation
import SplatIO

/// The projects of the library folder, the recent projects, and the actions on them. Every file operation runs
/// off the main actor in `LibraryFiles`; the store refreshes its list afterwards.
@Observable
@MainActor
final class LibraryStore {
    /// Packages of the library folder, unsorted. Recent projects outside the folder are in `outsideRecents`.
    private(set) var items: [LibraryItem] = []
    private(set) var outsideRecents: [LibraryItem] = []
    private(set) var isScanning: Bool = false
    /// Message for the user after a failed action. The view shows it in an alert and clears it.
    var message: String?
    /// Set by the New Project command. The library shows the sheet and clears the flag.
    var isNewProjectRequested: Bool = false

    /// Sizes by package URL, reused while a package's modification date is unchanged.
    @ObservationIgnored private var sizeCache: [URL: (modified: Date?, size: Int64)] = [:]
    /// Latest scan. A scan that finishes after a newer one started is dropped.
    @ObservationIgnored private var scanGeneration: Int = 0

    private static let lastOpenedKey: String = "libraryLastOpenedDates"
    private static let recentLimit: Int = 60

    /// Recent projects, most recently opened first: those in the folder that were opened, and those outside it.
    var recentItems: [LibraryItem] {
        let opened: [LibraryItem] = (items + outsideRecents)
            .filter { $0.lastOpened != nil }
            .sorted { (lhs: LibraryItem, rhs: LibraryItem) -> Bool in
                (lhs.lastOpened ?? .distantPast) > (rhs.lastOpened ?? .distantPast)
            }
        return Array(opened.prefix(Self.recentLimit))
    }

    // MARK: - Scanning

    /// Rescans `folder` and the recent documents. Sizes are computed for the packages whose modification date changed.
    func refresh(folder: URL) async {
        scanGeneration += 1
        let generation: Int = scanGeneration
        isScanning = true
        defer {
            if generation == scanGeneration {
                isScanning = false
            }
        }
        let folderItems: [LibraryItem]
        do {
            folderItems = try await LibraryFiles.scan(folder: folder)
        } catch {
            message = "The library folder could not be read: \(error.localizedDescription)"
            items = []
            outsideRecents = []
            return
        }
        let outside: [URL] = recentPackageURLs().filter { !Self.isInside($0, folder: folder) }
        let outsideItems: [LibraryItem] = await LibraryFiles.readItems(outside, outsideLibrary: true)
        guard generation == scanGeneration else {
            return
        }

        var withDates: [LibraryItem] = folderItems.map { (item: LibraryItem) -> LibraryItem in
            var result: LibraryItem = item
            result.lastOpened = lastOpened(for: item.url, folder: folder)
            return result
        }
        var outsideWithDates: [LibraryItem] = outsideItems.map { (item: LibraryItem) -> LibraryItem in
            var result: LibraryItem = item
            result.lastOpened = lastOpened(for: item.url, folder: folder)
            return result
        }
        items = withDates
        outsideRecents = outsideWithDates

        let sizes: [URL: Int64] = await sizesForChangedPackages(withDates + outsideWithDates)
        guard generation == scanGeneration else {
            return
        }
        withDates = withDates.map { (item: LibraryItem) -> LibraryItem in
            var result: LibraryItem = item
            result.sizeInBytes = sizes[item.url] ?? sizeCache[item.url]?.size
            return result
        }
        outsideWithDates = outsideWithDates.map { (item: LibraryItem) -> LibraryItem in
            var result: LibraryItem = item
            result.sizeInBytes = sizes[item.url] ?? sizeCache[item.url]?.size
            return result
        }
        items = withDates
        outsideRecents = outsideWithDates
    }

    /// Sizes of the packages whose modification date differs from the cached one. Unchanged packages reuse the cache.
    private func sizesForChangedPackages(_ all: [LibraryItem]) async -> [URL: Int64] {
        let changed: [URL] = all.filter { (item: LibraryItem) -> Bool in
            guard let cached = sizeCache[item.url] else {
                return true
            }
            return cached.modified != item.modificationDate
        }.map(\.url)
        guard !changed.isEmpty else {
            return [:]
        }
        let computed: [URL: Int64] = await LibraryFiles.byteSizes(of: changed)
        for item in all {
            if let size = computed[item.url] {
                sizeCache[item.url] = (item.modificationDate, size)
            }
        }
        return computed
    }

    // MARK: - Recents

    /// The `.gsproj` packages of the system's recent documents, most recent first.
    private func recentPackageURLs() -> [URL] {
        NSDocumentController.shared.recentDocumentURLs.filter {
            $0.pathExtension.lowercased() == LibraryNaming.packageExtension
        }
    }

    /// Last opened date of a package. The app's own record wins. A system recent document without a record gets a
    /// date from its position in the recent list, so the order is kept once written.
    private func lastOpened(for url: URL, folder: URL) -> Date? {
        let key: String = Self.recordKey(for: url, folder: folder)
        let stored: [String: Double] = UserDefaults.standard.dictionary(forKey: Self.lastOpenedKey) as? [String: Double] ?? [:]
        if let value = stored[key] {
            return Date(timeIntervalSince1970: value)
        }
        guard let index = recentPackageURLs().firstIndex(of: url) else {
            return nil
        }
        let date: Date = Date(timeIntervalSinceNow: -Double(index) * 60)
        var updated: [String: Double] = stored
        updated[key] = date.timeIntervalSince1970
        UserDefaults.standard.set(updated, forKey: Self.lastOpenedKey)
        return date
    }

    /// Records that `url` was opened now, in the app's record and in the system's recent documents.
    private func noteOpened(_ url: URL, folder: URL?) {
        if let folder {
            let key: String = Self.recordKey(for: url, folder: folder)
            var stored: [String: Double] = UserDefaults.standard.dictionary(forKey: Self.lastOpenedKey) as? [String: Double] ?? [:]
            stored[key] = Date().timeIntervalSince1970
            UserDefaults.standard.set(stored, forKey: Self.lastOpenedKey)
        }
        NSDocumentController.shared.noteNewRecentDocumentURL(url)
    }

    /// Empties the recent projects: the app's record and the system's recent documents.
    func clearRecents(folder: URL) async {
        UserDefaults.standard.removeObject(forKey: Self.lastOpenedKey)
        NSDocumentController.shared.clearRecentDocuments(nil)
        await refresh(folder: folder)
    }

    /// Key of a package in the app's record: its name inside the folder, its full path outside.
    private static func recordKey(for url: URL, folder: URL) -> String {
        if isInside(url, folder: folder) {
            return url.lastPathComponent
        }
        return url.path(percentEncoded: false)
    }

    private static func isInside(_ url: URL, folder: URL) -> Bool {
        url.deletingLastPathComponent().standardizedFileURL == folder.standardizedFileURL
    }

    // MARK: - Actions

    /// Opens each package as a document window. The library stays open.
    func open(_ urls: [URL], folder: URL) {
        for url in urls {
            NSDocumentController.shared.openDocument(withContentsOf: url, display: true) { [weak self] _, _, error in
                guard let self else {
                    return
                }
                if let error {
                    self.message = "The project could not be opened: \(error.localizedDescription)"
                }
            }
            noteOpened(url, folder: folder)
        }
        Task {
            await refresh(folder: folder)
        }
    }

    /// Creates an empty project named `name` in `folder` and opens it. Returns `false` when it failed.
    func createProject(named name: String, folder: URL) async -> Bool {
        do {
            let package: URL = try await LibraryFiles.createProject(named: name, in: folder)
            await refresh(folder: folder)
            open([package], folder: folder)
            return true
        } catch {
            message = Self.describe(error)
            return false
        }
    }

    /// Renames a package. Returns the new URL, or `nil` when it failed.
    func rename(_ url: URL, to name: String, folder: URL) async -> URL? {
        do {
            let renamed: URL = try await LibraryFiles.rename(url, to: name)
            await refresh(folder: folder)
            return renamed
        } catch {
            message = Self.describe(error)
            return nil
        }
    }

    /// Copies each package in `urls` next to it under a new name.
    func duplicate(_ urls: [URL], folder: URL) async {
        for url in urls {
            do {
                _ = try await LibraryFiles.duplicate(url)
            } catch {
                message = Self.describe(error)
            }
        }
        await refresh(folder: folder)
    }

    /// Moves packages to the Trash.
    func moveToTrash(_ urls: [URL], folder: URL) async {
        do {
            try await LibraryFiles.moveToTrash(urls)
        } catch {
            message = Self.describe(error)
        }
        await refresh(folder: folder)
    }

    /// Copies dropped packages into `folder`. Returns `false` when none was a package.
    func importDropped(_ urls: [URL], folder: URL) async -> Bool {
        var imported: Int = 0
        for url in urls where url.pathExtension.lowercased() == LibraryNaming.packageExtension {
            do {
                _ = try await LibraryFiles.importPackage(url, into: folder)
                imported += 1
            } catch {
                message = Self.describe(error)
            }
        }
        await refresh(folder: folder)
        return imported > 0
    }

    /// Shows the packages selected in the Finder.
    func reveal(_ urls: [URL]) {
        NSWorkspace.shared.activateFileViewerSelecting(urls)
    }

    static func describe(_ error: Error) -> String {
        if let libraryError = error as? LibraryFileError {
            return libraryError.localizedDescription
        }
        return "The operation failed: \(error.localizedDescription)"
    }
}
