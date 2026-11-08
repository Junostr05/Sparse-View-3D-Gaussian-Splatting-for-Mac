import Foundation
import Testing
import SplatCore

@Suite struct ExportGeometryTests {
    @Test func sizeKeepsTheAspectAndRoundsToEvenNumbers() {
        #expect(ExportGeometry.size(width: 1920, aspect: 16.0 / 9.0) == PixelSize(width: 1920, height: 1080))
        #expect(ExportGeometry.size(width: 3840, aspect: 16.0 / 9.0) == PixelSize(width: 3840, height: 2160))
        // 1281 wide: the width rounds down to 1280, and 1280 / (4 / 3) = 960 is already even.
        #expect(ExportGeometry.size(width: 1281, aspect: 4.0 / 3.0) == PixelSize(width: 1280, height: 960))
        // An odd height from the aspect ratio rounds down to the even number below it.
        #expect(ExportGeometry.size(width: 1000, aspect: 1000.0 / 701.0) == PixelSize(width: 1000, height: 700))
    }

    @Test func sizeFallsBackToSixteenByNineForABadAspect() {
        #expect(ExportGeometry.size(width: 1920, aspect: 0) == PixelSize(width: 1920, height: 1080))
        #expect(ExportGeometry.size(width: 1920, aspect: .nan) == PixelSize(width: 1920, height: 1080))
    }

    @Test func evenFloorIsAtLeastTwo() {
        #expect(ExportGeometry.evenFloor(0) == 2)
        #expect(ExportGeometry.evenFloor(1.9) == 2)
        #expect(ExportGeometry.evenFloor(7) == 6)
        #expect(ExportGeometry.evenFloor(-5) == 2)
        #expect(ExportGeometry.evenFloor(.infinity) == 2)
    }

    @Test func validateAcceptsTheLimitAndRefusesLargerOrEmptySizes() {
        #expect(ExportGeometry.validate(PixelSize(width: 1920, height: 1080)) == nil)
        #expect(ExportGeometry.validate(PixelSize(width: 8192, height: 8192)) == nil)
        #expect(ExportGeometry.validate(PixelSize(width: 8194, height: 2)) == .tooLarge(maxSide: 8192, maxPixels: 8192 * 8192))
        #expect(ExportGeometry.validate(PixelSize(width: 0, height: 1080)) == .empty)
        #expect(ExportGeometry.validate(PixelSize(width: 1920, height: -2)) == .empty)
    }

    @Test func frameCountIsSecondsTimesRateWithATwoFrameMinimum() {
        #expect(ExportGeometry.frameCount(seconds: 2, fps: 30) == 60)
        #expect(ExportGeometry.frameCount(seconds: 4, fps: 30) == 120)
        #expect(ExportGeometry.frameCount(seconds: 60, fps: 30) == 1800)
        #expect(ExportGeometry.frameCount(seconds: 0.01, fps: 30) == 2)
        #expect(ExportGeometry.frameCount(seconds: 10.0 / 30.0, fps: 30) == 10)
    }

    @Test func defaultDurationIsTwoSecondsPerCameraWithinTheRange() {
        #expect(ExportGeometry.defaultVideoDuration(cameraCount: 0) == ExportGeometry.turntableSeconds)
        #expect(ExportGeometry.defaultVideoDuration(cameraCount: 1) == 4)
        #expect(ExportGeometry.defaultVideoDuration(cameraCount: 3) == 6)
        #expect(ExportGeometry.defaultVideoDuration(cameraCount: 40) == 60)
        #expect(ExportGeometry.clampedDuration(1) == 4)
        #expect(ExportGeometry.clampedDuration(90) == 60)
        #expect(ExportGeometry.clampedDuration(.nan) == 4)
    }

    @Test func memoryCheckUsesAQuarterOfPhysicalMemory() {
        let eightK = PixelSize(width: 8192, height: 8192)
        // Three 8192² BGRA frames take about 805 MB.
        #expect(ExportGeometry.byteCount(of: eightK, copies: 3) == 8192 * 8192 * 4 * 3)
        #expect(ExportGeometry.fitsInMemory(eightK, copies: 3, physicalMemory: 8 << 30))
        #expect(!ExportGeometry.fitsInMemory(eightK, copies: 3, physicalMemory: 1 << 30))
        #expect(ExportGeometry.fitsInMemory(PixelSize(width: 1920, height: 1080), copies: 2, physicalMemory: 8 << 30))
    }
}
