import Testing
import SplatCore

@Suite struct CommandSearchTests {
    private struct Item: CommandSearchable {
        let title: String
        var keywords: [String] = []
        var isEnabled: Bool = true
    }

    private func titles(_ query: String, _ items: [Item]) -> [String] {
        CommandSearch.rank(query: query, commands: items).map { $0.title }
    }

    private let sample: [Item] = [
        Item(title: "Frame All", keywords: ["fit", "home"]),
        Item(title: "Show Grid"),
        Item(title: "Show Axes"),
        Item(title: "Export Flight Video"),
        Item(title: "Save Image"),
        Item(title: "Undo"),
    ]

    @Test func emptyQueryKeepsRegistryOrderOfEnabledCommands() {
        var items = sample
        items[1].isEnabled = false
        #expect(titles("", items) == ["Frame All", "Show Axes", "Export Flight Video", "Save Image", "Undo"])
        #expect(titles("   ", items) == titles("", items))
    }

    @Test func subsequenceMatchesAndNonMatchesAreDropped() {
        // "fv" is a subsequence of "Flight Video" (f, then v) and of nothing else in the sample.
        #expect(titles("fv", sample) == ["Export Flight Video"])
        #expect(titles("xyz", sample).isEmpty)
        // Characters must appear in order.
        #expect(titles("dla", sample).isEmpty)
    }

    @Test func prefixOfTitleRanksAboveMiddleMatch() {
        let items: [Item] = [Item(title: "Undo"), Item(title: "Redo Undo Step")]
        #expect(titles("und", items) == ["Undo", "Redo Undo Step"])
    }

    @Test func wordStartRanksAboveInsideWord() {
        // "ax" starts the word "Axes" in one title and sits inside "Taxonomy" in the other.
        let items: [Item] = [Item(title: "Taxonomy"), Item(title: "Show Axes")]
        #expect(titles("ax", items) == ["Show Axes", "Taxonomy"])
    }

    @Test func matchingIsCaseInsensitive() {
        #expect(titles("FRAME", sample) == ["Frame All"])
        #expect(titles("frame all", sample) == ["Frame All"])
    }

    @Test func matchingIgnoresDiacritics() {
        let items: [Item] = [Item(title: "Émigré Export"), Item(title: "Undo")]
        #expect(titles("emigre", items) == ["Émigré Export"])
        #expect(titles("ÉMIGRÉ", items) == ["Émigré Export"])
    }

    @Test func keywordsMatchButRankBelowTitles() {
        // "fit" is a keyword of "Frame All" and also a loose subsequence of "Export Flight Video" (f-l-i-g-h-t).
        // The keyword match is the stronger one, so it comes first.
        #expect(titles("fit", sample) == ["Frame All", "Export Flight Video"])
        let items: [Item] = [Item(title: "Frame Selection", keywords: ["fit"]), Item(title: "Fit Window")]
        #expect(titles("fit", items) == ["Fit Window", "Frame Selection"])
    }

    @Test func disabledCommandsAreFilteredOut() {
        var items = sample
        items[0].isEnabled = false
        #expect(!titles("frame", items).contains("Frame All"))
        #expect(!titles("", items).contains("Frame All"))
    }

    @Test func tiesKeepRegistryOrder() {
        let items: [Item] = [Item(title: "Show Grid"), Item(title: "Show Grid")]
        let ranked: [Item] = CommandSearch.rank(query: "grid", commands: items)
        #expect(ranked.count == 2)
        // Identical items: the sort is stable, so the first one stays first.
        #expect(ranked.map { $0.title } == ["Show Grid", "Show Grid"])
        let mixed: [Item] = [Item(title: "Axes Grid"), Item(title: "Grid Axes")]
        #expect(titles("grid", mixed) == ["Grid Axes", "Axes Grid"])
    }

    @Test func wholeTitlePrefixOutranksScatteredWords() {
        let items: [Item] = [Item(title: "Save Image"), Item(title: "Show Axes Image")]
        #expect(titles("save image", items) == ["Save Image"])
    }
}
