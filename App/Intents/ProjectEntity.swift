import AppIntents
import Foundation
import SplatIO

/// A project of the library, as Shortcuts and Spotlight see it. The identifier is the package path.
struct ProjectEntity: AppEntity, Identifiable {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "SparseSplat Project"
    static let defaultQuery: ProjectEntityQuery = ProjectEntityQuery()

    let id: String
    let name: String

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(name)")
    }

    var url: URL {
        URL(fileURLWithPath: id, isDirectory: true)
    }

    init(path: String, name: String) {
        self.id = path
        self.name = name
    }

    init(item: LibraryItem) {
        self.init(path: item.url.path(percentEncoded: false), name: item.name)
    }
}

/// Lists the projects of the library folder. Without a library folder the list is empty.
@MainActor
struct ProjectEntityQuery: EntityStringQuery {
    func entities(for identifiers: [String]) async throws -> [ProjectEntity] {
        try await allProjects().filter { identifiers.contains($0.id) }
    }

    func entities(matching string: String) async throws -> [ProjectEntity] {
        try await allProjects().filter { $0.name.localizedCaseInsensitiveContains(string) }
    }

    func suggestedEntities() async throws -> [ProjectEntity] {
        try await allProjects()
    }

    private func allProjects() async throws -> [ProjectEntity] {
        guard let folder: URL = LibraryFolder().readyURL else {
            return []
        }
        let items: [LibraryItem] = try await IntentWork.projects(in: folder)
        return items.map { ProjectEntity(item: $0) }
    }
}
