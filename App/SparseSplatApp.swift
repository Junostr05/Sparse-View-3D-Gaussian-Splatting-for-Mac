import SwiftUI

@main
struct SparseSplatApp: App {
    init() {
        // By default a document app shows an Open panel (filtered to document types) at launch. This
        // app should open an empty project window instead, where captures are imported from folders.
        UserDefaults.standard.register(defaults: ["NSShowAppCentricOpenPanelInsteadOfUntitledFile": false])
    }

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
