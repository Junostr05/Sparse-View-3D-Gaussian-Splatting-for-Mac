import Testing
@testable import SplatIO

@Suite struct CaptureSamplingTests {
    @Test func candidateTimesAreMidpointsBelowDuration() {
        let times: [Double] = CaptureSampling.candidateTimes(duration: 1, interval: 0.25, maxCount: 100)
        #expect(times == [0.125, 0.375, 0.625, 0.875])
    }

    @Test func candidateTimesExcludeTimesAtOrAfterDuration() {
        // 0.5 * 0.25 = 0.125 < 0.2, 0.375 >= 0.2 -> only one candidate.
        let times: [Double] = CaptureSampling.candidateTimes(duration: 0.2, interval: 0.25, maxCount: 100)
        #expect(times == [0.125])
    }

    @Test func candidateTimesEmptyForNonPositiveInputs() {
        #expect(CaptureSampling.candidateTimes(duration: 0, interval: 0.25, maxCount: 10).isEmpty)
        #expect(CaptureSampling.candidateTimes(duration: -3, interval: 0.25, maxCount: 10).isEmpty)
        #expect(CaptureSampling.candidateTimes(duration: 3, interval: 0, maxCount: 10).isEmpty)
        #expect(CaptureSampling.candidateTimes(duration: 3, interval: -1, maxCount: 10).isEmpty)
        #expect(CaptureSampling.candidateTimes(duration: 3, interval: 0.25, maxCount: 0).isEmpty)
        #expect(CaptureSampling.candidateTimes(duration: .nan, interval: 0.25, maxCount: 10).isEmpty)
    }

    @Test func candidateTimesThinnedEvenly() {
        // 10 s at 0.25 s gives 40 candidates; keeping 4 takes the middle of each group of 10.
        let times: [Double] = CaptureSampling.candidateTimes(duration: 10, interval: 0.25, maxCount: 4)
        #expect(times == [1.375, 3.875, 6.375, 8.875])
    }

    @Test func thinnedTimesAreAscendingUniqueAndWithinDuration() {
        let times: [Double] = CaptureSampling.candidateTimes(duration: 37.3, interval: 0.1, maxCount: 50)
        #expect(times.count == 50)
        for (previous, next) in zip(times, times.dropFirst()) {
            #expect(previous < next)
        }
        #expect((times.last ?? 0) < 37.3)
    }

    @Test func noThinningWhenWithinMaxCount() {
        let times: [Double] = CaptureSampling.candidateTimes(duration: 3, interval: 0.25, maxCount: 240)
        #expect(times.count == 12)
        #expect(times.first == 0.125)
        #expect(times.last == 2.875)
    }

    @Test func frameNamingPadsToFourDigits() {
        #expect(FrameNaming.name(prefix: "frame", index: 1, ext: "png") == "frame_0001.png")
        #expect(FrameNaming.name(prefix: "photo", index: 12, ext: "jpg") == "photo_0012.jpg")
        #expect(FrameNaming.name(prefix: "frame", index: 12345, ext: "png") == "frame_12345.png")
    }

    @Test func frameNamingIgnoresLeadingDotInExtension() {
        #expect(FrameNaming.name(prefix: "frame", index: 7, ext: ".png") == "frame_0007.png")
    }
}
