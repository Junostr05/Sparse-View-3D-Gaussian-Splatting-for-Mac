import Foundation

/// Creates package contents from folders on disk, without a document. Mirrors the app's COLMAP import.
public enum ProjectFolderImport {
    /// Reads `sparse/0/{cameras,images,points3D}.bin` and the images named by the COLMAP records from `folder/images/`.
    /// Training settings of `contents` are kept, as in `ProjectBundle.importCOLMAP`.
    public static func colmap(folder: URL, into contents: ProjectContents) throws -> ProjectContents {
        let sparse: URL = folder.appending(path: "sparse/0", directoryHint: .isDirectory)
        let cameras: Data
        let images: Data
        do {
            cameras = try Data(contentsOf: sparse.appending(path: "cameras.bin"))
            images = try Data(contentsOf: sparse.appending(path: "images.bin"))
        } catch {
            throw DatasetError.truncated(file: "sparse/0 (cameras.bin and images.bin)")
        }
        let points: Data? = try? Data(contentsOf: sparse.appending(path: "points3D.bin"))
        let records: [COLMAPImage] = try COLMAPReader.readImages(images)
        let imageRoot: URL = folder.appending(path: "images", directoryHint: .isDirectory)
        var imageFiles: [String: Data] = [:]
        var missing: [String] = []
        for record in records {
            if let data = try? Data(contentsOf: imageRoot.appending(path: record.name)) {
                imageFiles[record.name] = data
            } else {
                missing.append(record.name)
            }
        }
        guard missing.isEmpty else {
            throw DatasetError.missingImages(missing)
        }
        return ProjectBundle.importCOLMAP(cameras: cameras,
                                          images: images,
                                          points3D: points,
                                          imageFiles: imageFiles,
                                          into: contents)
    }
}
