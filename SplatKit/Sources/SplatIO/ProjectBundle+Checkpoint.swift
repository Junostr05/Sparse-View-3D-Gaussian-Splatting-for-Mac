import SplatCore

extension ProjectBundle {
    /// The latest checkpoint without its stored deletions: the cloud a continuation starts from and the cloud an
    /// export writes. The deletions apply only while they belong to that checkpoint. `nil` without a checkpoint.
    public static func checkpointWithoutDeletions(_ contents: ProjectContents) throws -> GaussianCloud? {
        guard let checkpoint = try latestCheckpoint(contents) else {
            return nil
        }
        guard let stored = try deletions(contents),
              stored.baseGaussianCount == checkpoint.count else {
            return checkpoint
        }
        let mask: SelectionMask = SelectionMask(count: checkpoint.count, indices: stored.deleted.map { Int($0) })
        return try checkpoint.removing(mask)
    }
}
