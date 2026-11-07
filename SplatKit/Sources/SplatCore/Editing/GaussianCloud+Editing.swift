extension GaussianCloud {
    /// Returns a copy without the Gaussians selected in `mask`.
    ///
    /// Survivors keep their relative order in every array. Removing every Gaussian yields an
    /// empty cloud with the same SH degree. A mask whose count differs from this cloud's count
    /// throws `.inconsistentLengths`.
    public func removing(_ mask: SelectionMask) throws(GaussianCloudError) -> GaussianCloud {
        guard mask.count == count else {
            throw .inconsistentLengths
        }
        let shStride: Int = Self.coefficientCount(degree: shDegree) * 3

        var keptCount: Int = 0
        for index in 0..<count where !mask[index] {
            keptCount += 1
        }

        var keptPositions: [Float] = []
        var keptLogScales: [Float] = []
        var keptRotations: [Float] = []
        var keptOpacity: [Float] = []
        var keptSH: [Float] = []
        keptPositions.reserveCapacity(3 * keptCount)
        keptLogScales.reserveCapacity(3 * keptCount)
        keptRotations.reserveCapacity(4 * keptCount)
        keptOpacity.reserveCapacity(keptCount)
        keptSH.reserveCapacity(shStride * keptCount)

        for index in 0..<count where !mask[index] {
            let positionBase: Int = 3 * index
            keptPositions.append(positions[positionBase])
            keptPositions.append(positions[positionBase + 1])
            keptPositions.append(positions[positionBase + 2])

            keptLogScales.append(logScales[positionBase])
            keptLogScales.append(logScales[positionBase + 1])
            keptLogScales.append(logScales[positionBase + 2])

            let rotationBase: Int = 4 * index
            keptRotations.append(rotations[rotationBase])
            keptRotations.append(rotations[rotationBase + 1])
            keptRotations.append(rotations[rotationBase + 2])
            keptRotations.append(rotations[rotationBase + 3])

            keptOpacity.append(opacityLogits[index])

            let shBase: Int = index * shStride
            keptSH.append(contentsOf: sh[shBase..<shBase + shStride])
        }

        return GaussianCloud(
            validated: shDegree,
            positions: keptPositions,
            logScales: keptLogScales,
            rotations: keptRotations,
            opacityLogits: keptOpacity,
            sh: keptSH
        )
    }
}
