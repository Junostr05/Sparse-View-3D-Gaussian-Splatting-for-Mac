import SwiftUI

@main
struct SparseSplatApp: App {
    var body: some Scene {
        DocumentGroup(viewing: SplatDocument.self) { file in
            SceneWindow(scene: file.document.scene)
        }
    }
}
