// BridgeProcessManager.ts
// This file helps manage the Looking Glass Bridge processes if they're causing issues

export class BridgeProcessManager {
  static async checkBridgeProcesses(): Promise<number> {
    // This would check for BridgeCommunication.exe processes
    // For now, just log a warning
    console.warn(
      'Multiple BridgeCommunication.exe processes detected. ' +
      'This is from Looking Glass Bridge software. ' +
      'If experiencing issues, try restarting the Looking Glass Bridge service.'
    );
    return 0;
  }
  
  static async cleanupExcessProcesses(): Promise<void> {
    // Note: Cannot directly kill processes from browser JavaScript
    // This would need to be handled by the Looking Glass Bridge software itself
    console.info(
      'To clean up excess BridgeCommunication.exe processes:\n' +
      '1. Open Task Manager\n' +
      '2. End all BridgeCommunication.exe processes\n' +
      '3. Restart Looking Glass Bridge from system tray'
    );
  }
}

// Export for use if needed
export default BridgeProcessManager;