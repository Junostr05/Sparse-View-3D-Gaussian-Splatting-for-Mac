import Foundation
import UserNotifications

/// Notifications of the training queue. Permission is requested once, when the first job is added, never at launch.
@MainActor
enum QueueNotifications {
    private static let requestedKey: String = "trainingQueueNotificationsRequested"

    /// Asks for permission the first time a job is queued. Later calls do nothing.
    static func requestPermissionOnce() {
        let defaults: UserDefaults = UserDefaults.standard
        guard !defaults.bool(forKey: requestedKey) else {
            return
        }
        defaults.set(true, forKey: requestedKey)
        Task {
            _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
        }
    }

    /// Posts a notification when the user allowed them. Without permission nothing is shown.
    static func post(title: String, body: String) {
        Task {
            let center: UNUserNotificationCenter = UNUserNotificationCenter.current()
            let settings: UNNotificationSettings = await center.notificationSettings()
            guard settings.authorizationStatus == .authorized else {
                return
            }
            let content: UNMutableNotificationContent = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            let request: UNNotificationRequest = UNNotificationRequest(identifier: UUID().uuidString,
                                                                       content: content,
                                                                       trigger: nil)
            try? await center.add(request)
        }
    }
}
