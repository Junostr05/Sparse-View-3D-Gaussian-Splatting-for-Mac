import Testing
import SplatCore

// MARK: - Helpers

/// Builds a grey image from a per-pixel value function (explicit loops keep the type checker happy).
func makeGreyImage(width: Int, height: Int, value: (Int, Int) -> Float) -> RGBImage {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for y in 0..<height {
        for x in 0..<width {
            let v: Float = value(x, y)
            pixels.append(SIMD3<Float>(v, v, v))
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

/// 3x3 box blur over the interior; the border is copied unchanged.
func boxBlurred(_ image: RGBImage) -> RGBImage {
    let w: Int = image.width
    let h: Int = image.height
    var out: [SIMD3<Float>] = image.pixels
    for y in 1..<(h - 1) {
        for x in 1..<(w - 1) {
            var acc: SIMD3<Float> = .zero
            for dy in -1...1 {
                for dx in -1...1 {
                    let p: SIMD3<Float> = image.pixels[(y + dy) * w + (x + dx)]
                    acc += p
                }
            }
            out[y * w + x] = acc / Float(9)
        }
    }
    return RGBImage(width: w, height: h, pixels: out)
}

/// Candidates whose index is their position in `times` / `sharpness`.
func makeCandidates(times: [Double], sharpness: [Double]) -> [FrameCandidate] {
    var result: [FrameCandidate] = []
    for i in 0..<times.count {
        result.append(FrameCandidate(index: i, time: times[i], sharpness: sharpness[i]))
    }
    return result
}

// MARK: - FrameSharpness

@Test func sharpImageHasHigherLaplacianVarianceThanBlurred() {
    let sharp: RGBImage = makeGreyImage(width: 8, height: 8) { x, y in
        (x + y) % 2 == 0 ? 1 : 0
    }
    let blurred: RGBImage = boxBlurred(sharp)
    let sharpVariance: Double = FrameSharpness.laplacianVariance(sharp)
    let blurredVariance: Double = FrameSharpness.laplacianVariance(blurred)
    #expect(sharpVariance > 0)
    #expect(sharpVariance > blurredVariance)
}

@Test func flatImageHasZeroVariance() {
    let flat: RGBImage = makeGreyImage(width: 5, height: 5) { _, _ in 0.4 }
    #expect(abs(FrameSharpness.laplacianVariance(flat)) < 1e-12)
}

@Test func tinyImageReturnsZero() {
    let tiny: RGBImage = makeGreyImage(width: 2, height: 2) { x, y in
        Float(x + y) / 3
    }
    #expect(FrameSharpness.laplacianVariance(tiny) == 0)
}

// MARK: - FrameSelector: basic behaviour

@Test func selectsOneSharpestFramePerBin() {
    // 12 candidates over 0...11 s, count 4 -> bins [0,2.75), [2.75,5.5), [5.5,8.25), [8.25,11].
    let times: [Double] = (0..<12).map { Double($0) }
    let sharpness: [Double] = [0.1, 0.9, 0.2, 0.3, 0.8, 0.4, 0.95, 0.5, 0.6, 0.2, 0.3, 0.7]
    let result: [Int] = FrameSelector.select(makeCandidates(times: times, sharpness: sharpness), count: 4)
    #expect(result == [1, 4, 6, 11])
}

@Test func resultIsSortedByTimeAndUnique() {
    // Input deliberately out of time order.
    let candidates: [FrameCandidate] = [
        FrameCandidate(index: 7, time: 6.0, sharpness: 0.4),
        FrameCandidate(index: 2, time: 0.5, sharpness: 0.9),
        FrameCandidate(index: 9, time: 9.5, sharpness: 0.8),
        FrameCandidate(index: 4, time: 3.0, sharpness: 0.7),
        FrameCandidate(index: 1, time: 1.5, sharpness: 0.2),
        FrameCandidate(index: 5, time: 4.5, sharpness: 0.6),
        FrameCandidate(index: 8, time: 8.0, sharpness: 0.5),
        FrameCandidate(index: 0, time: 0.0, sharpness: 0.3),
        FrameCandidate(index: 3, time: 2.0, sharpness: 0.1),
        FrameCandidate(index: 6, time: 7.0, sharpness: 0.65),
    ]
    let result: [Int] = FrameSelector.select(candidates, count: 5)

    #expect(result.count == 5)
    #expect(Set(result).count == result.count)

    var lastTime: Double = -Double.infinity
    for index in result {
        let time: Double = candidates.first { $0.index == index }!.time
        #expect(time >= lastTime)
        lastTime = time
    }
}

@Test func countLargerThanCandidatesReturnsAll() {
    let candidates: [FrameCandidate] = [
        FrameCandidate(index: 7, time: 2.0, sharpness: 0.1),
        FrameCandidate(index: 3, time: 0.0, sharpness: 0.9),
        FrameCandidate(index: 5, time: 1.0, sharpness: 0.5),
    ]
    #expect(FrameSelector.select(candidates, count: 10) == [3, 5, 7])
    #expect(FrameSelector.select(candidates, count: 3) == [3, 5, 7])
}

@Test func countZeroOrEmptyReturnsEmpty() {
    let candidates: [FrameCandidate] = [
        FrameCandidate(index: 0, time: 0.0, sharpness: 0.5),
        FrameCandidate(index: 1, time: 1.0, sharpness: 0.5),
    ]
    #expect(FrameSelector.select(candidates, count: 0).isEmpty)
    #expect(FrameSelector.select(candidates, count: -2).isEmpty)
    #expect(FrameSelector.select([], count: 4).isEmpty)
}

@Test func emptyBinUsesNearestCandidate() {
    // Span 0...10, count 3 -> bins [0,3.33), [3.33,6.67), [6.67,10]. The middle bin is empty.
    // Centre of the empty bin is 5.0; nearest unchosen candidate is t=2 (distance 3) before t=9 (distance 4).
    let times: [Double] = [0.0, 1.0, 2.0, 9.0, 10.0]
    let sharpness: [Double] = [0.5, 0.6, 0.4, 0.3, 0.9]
    let result: [Int] = FrameSelector.select(makeCandidates(times: times, sharpness: sharpness), count: 3)
    #expect(result == [1, 2, 4])
}

// MARK: - FrameSelector: minimum spacing

@Test func minimumSpacingIsRespected() {
    // Sharp cluster at 4, 5, 6. Bins [0,3), [3,6), [6,9].
    // Bin 0 -> 1, bin 1 -> 4 (fits), bin 2 -> 6 violates spacing 3 (distance 2), so 7 replaces it.
    let times: [Double] = (0..<10).map { Double($0) }
    let sharpness: [Double] = [0.2, 0.3, 0.1, 0.6, 1.0, 0.9, 0.95, 0.5, 0.4, 0.45]
    let result: [Int] = FrameSelector.select(
        makeCandidates(times: times, sharpness: sharpness),
        count: 3,
        minimumSpacing: 3.0
    )
    #expect(result == [1, 4, 7])

    // Without spacing the same input picks the cluster member 6.
    let unconstrained: [Int] = FrameSelector.select(makeCandidates(times: times, sharpness: sharpness), count: 3)
    #expect(unconstrained == [1, 4, 6])
}

@Test func backfillFillsUpWhenBinIsSkipped() {
    // Bins [0,3), [3,6), [6,9]. Bin 1 is skipped: both candidates lie within 3 s of the bin 0 pick (t=1).
    // Bin 2 picks t=6. Backfill then takes the sharpest unchosen candidate that fits: t=9 (t=0, 3.5, 3.8 do not fit).
    let times: [Double] = [0.0, 1.0, 3.5, 3.8, 6.0, 9.0]
    let sharpness: [Double] = [0.5, 0.9, 0.8, 0.7, 0.6, 0.3]
    let result: [Int] = FrameSelector.select(
        makeCandidates(times: times, sharpness: sharpness),
        count: 3,
        minimumSpacing: 3.0
    )
    #expect(result == [1, 4, 5])

    let chosenTimes: [Double] = result.map { times[$0] }
    for i in 0..<chosenTimes.count {
        for j in (i + 1)..<chosenTimes.count {
            #expect(abs(chosenTimes[i] - chosenTimes[j]) >= 3.0)
        }
    }
}

// MARK: - FrameSelector: determinism and degenerate input

@Test func isDeterministicForTies() {
    // All sharpness equal: the earliest candidate in each bin wins.
    let times: [Double] = (0..<9).map { Double($0) }
    let sharpness: [Double] = Array(repeating: 0.5, count: 9)
    let forward: [FrameCandidate] = makeCandidates(times: times, sharpness: sharpness)
    #expect(FrameSelector.select(forward, count: 3) == [0, 3, 6])

    // Input order must not matter.
    let reversed: [FrameCandidate] = Array(forward.reversed())
    #expect(FrameSelector.select(reversed, count: 3) == [0, 3, 6])
}

@Test func degenerateTimeSpanDoesNotCrash() {
    // All candidates share one timestamp: everything lands in bin 0, the other bin is empty.
    let times: [Double] = [2.0, 2.0, 2.0, 2.0]
    let sharpness: [Double] = [0.1, 0.2, 0.9, 0.3]
    let result: [Int] = FrameSelector.select(makeCandidates(times: times, sharpness: sharpness), count: 2)
    #expect(result == [0, 2])
}
