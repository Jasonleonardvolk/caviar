declare namespace App {
  interface Locals {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
  interface PageData {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
}