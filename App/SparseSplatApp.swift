import SwiftUI

@main
struct SparseSplatApp: App {
    var body: some Scene {
        // First group: launching the app opens an empty project. Keep it first.
        DocumentGroup(newDocument: ProjectDocument()) { file in
            ProjectWindow(document: file.$document)
        }
        DocumentGroup(viewing: SplatDocument.self) { file in
            SceneWindow(scene: file.document.scene)
        }
    }
}
