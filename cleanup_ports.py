#!/usr/bin/env python3
# cleanup_ports.py

from port_manager import PortManager

def main():
    print("🧹 TORI Port Cleanup Tool")
    print("=" * 50)
    pm = PortManager()
    services = {
        "audio_bridge": 8765,
        "concept_mesh_bridge": 8766,
        "api_server": 8002,
        "mcp_server": 8100,
        "frontend": 5173,
    }
    cleaned = 0
    for name, port in services.items():
        if not pm.is_port_free(port):
            print(f"🔫 Freeing {name} on port {port}…")
            if pm.force_free_port(port):
                print(f"✅ Port {port} freed")
                cleaned += 1
            else:
                print(f"❌ Failed to free port {port}")
        else:
            print(f"✅ Port {port} already free")
    print(f"\n🎉 Cleaned {cleaned} ports")
    if input("\nClear saved config? (y/N): ").lower() == 'y':
        pm.allocated_ports = {}
        pm._save_config()
        print("✅ Config cleared")

if __name__ == "__main__":
    main()
