import SwiftUI

@main
struct SparseSplatCompanionApp: App {
    @State private var model = CompanionModel()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 16) {
                Text("SparseSplat Companion")
                    .font(.title2.bold())
                Text(model.status)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button("Start") {
                    model.start()
                }
                .buttonStyle(.borderedProminent)
                Button("Stop") {
                    model.stop()
                }
                .buttonStyle(.bordered)
            }
            .padding()
            .onAppear {
                model.start()
            }
        }
    }
}
