#!/usr/bin/env python3
"""
ZeroMQ Key Rotation Script
==========================

Rotates ZeroMQ encryption keys and broadcasts update events.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from zmq_key_rotation import ZMQKeyManager, ZMQKeyRotationService
from communication import CommunicationFabric
from dickbox_config import DickboxConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main rotation function"""
    try:
        # Load configuration
        config = DickboxConfig.from_env()
        
        # Create key manager
        key_manager = ZMQKeyManager(Path("/etc/tori/zmq_keys"))
        
        # Create communication fabric for broadcasting
        comm_config = {
            "enable_zeromq": config.enable_zeromq,
            "zeromq_pub_port": config.zeromq_pub_port
        }
        
        fabric = CommunicationFabric(comm_config)
        await fabric.start()
        
        # Create rotation service
        rotation_service = ZMQKeyRotationService(key_manager, fabric)
        
        # Rotate keys and broadcast
        new_keys = await rotation_service.rotate_and_broadcast()
        
        logger.info(f"Successfully rotated keys: {new_keys['timestamp']}")
        logger.info(f"Public key: {new_keys['public_key']}")
        
        # Give time for broadcast to propagate
        await asyncio.sleep(2)
        
        # Stop fabric
        await fabric.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Key rotation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
