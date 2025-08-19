// Simple logger utility

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4
}

class Logger {
  private level: LogLevel = LogLevel.INFO;
  private prefix: string = '[TORI]';
  
  setLevel(level: LogLevel): void {
    this.level = level;
  }
  
  setPrefix(prefix: string): void {
    this.prefix = prefix;
  }
  
  debug(...args: any[]): void {
    if (this.level <= LogLevel.DEBUG) {
      console.log(`${this.prefix}[DEBUG]`, ...args);
    }
  }
  
  info(...args: any[]): void {
    if (this.level <= LogLevel.INFO) {
      console.log(`${this.prefix}[INFO]`, ...args);
    }
  }
  
  warn(...args: any[]): void {
    if (this.level <= LogLevel.WARN) {
      console.warn(`${this.prefix}[WARN]`, ...args);
    }
  }
  
  error(...args: any[]): void {
    if (this.level <= LogLevel.ERROR) {
      console.error(`${this.prefix}[ERROR]`, ...args);
    }
  }
  
  time(label: string): void {
    if (this.level <= LogLevel.DEBUG) {
      console.time(`${this.prefix} ${label}`);
    }
  }
  
  timeEnd(label: string): void {
    if (this.level <= LogLevel.DEBUG) {
      console.timeEnd(`${this.prefix} ${label}`);
    }
  }
  
  group(label: string): void {
    if (this.level <= LogLevel.DEBUG) {
      console.group(`${this.prefix} ${label}`);
    }
  }
  
  groupEnd(): void {
    if (this.level <= LogLevel.DEBUG) {
      console.groupEnd();
    }
  }
}

// Export singleton instance
export const logger = new Logger();

// Export for testing or multiple instances
export { Logger };

// Convenience exports
export const log = logger.info.bind(logger);
export const debug = logger.debug.bind(logger);
export const warn = logger.warn.bind(logger);
export const error = logger.error.bind(logger);
