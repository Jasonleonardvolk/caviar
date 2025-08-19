/**
 * Delta Protocol Resilience Test
 * 
 * Tests the Delta Protocol under various network conditions:
 * - 5% packet loss
 * - 200ms network jitter
 * - Out of order packets
 * - Sequence number rollover
 * 
 * Usage: node test_delta_protocol.js
 */

const { DeltaEncoder, DeltaDecoder } = require('./packages/runtime-bridge/src/DeltaProtocol');

// Statistics tracking
const stats = {
  packetsTotal: 0,
  packetsSent: 0,
  packetsReceived: 0,
  packetsLost: 0,
  resyncRequests: 0,
  ackTotal: 0,
  ackReceived: 0,
  maxLatency: 0,
  latencySum: 0,
  testStartTime: Date.now(),
};

// Test parameters
const TEST_DURATION_SEC = 30;
const PACKET_RATE_HZ = 500;    // 500 Hz packet rate
const PACKET_LOSS_RATE = 0.05; // 5% packet loss
const MAX_JITTER_MS = 200;     // Up to 200ms jitter

// Create encoder and decoder
const encoder = new DeltaEncoder({
  maxHistory: 20,
  requireAck: true,
  ackTimeout: 300,  // 300ms timeout (aggressive for testing)
  backoffFactor: 1.5,
  maxBackoff: 5000, // 5 seconds max backoff
});

const decoder = new DeltaDecoder({
  onResyncNeeded: () => {
    stats.resyncRequests++;
    console.log(`[${formatTime()}] Resync requested`);
  }
});

// Packet queue (simulates network)
const networkQueue = [];

/**
 * Format timestamp as [MM:SS.mmm]
 */
function formatTime() {
  const elapsed = Date.now() - stats.testStartTime;
  const minutes = Math.floor(elapsed / 60000);
  const seconds = Math.floor((elapsed % 60000) / 1000);
  const millis = elapsed % 1000;
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
}

/**
 * Delay function with random jitter
 * 
 * @param {function} callback Function to call after delay
 */
function delayWithJitter(callback) {
  // Random delay between 0 and MAX_JITTER_MS
  const jitter = Math.random() * MAX_JITTER_MS;
  setTimeout(callback, jitter);
}

/**
 * Process network queue
 * Simulates packet delivery with loss and jitter
 */
function processNetworkQueue() {
  // Process all packets in queue, might be delivered out of order due to jitter
  const packetsToProcess = [...networkQueue];
  networkQueue.length = 0;
  
  for (const { packet, timestamp, type, sequence } of packetsToProcess) {
    stats.packetsTotal++;
    
    // Simulate packet loss
    if (Math.random() < PACKET_LOSS_RATE) {
      stats.packetsLost++;
      if (type === 'data') {
        // Log only for data packets, not ACKs
        console.log(`[${formatTime()}] Packet ${sequence} lost in transit`);
      }
      continue;
    }
    
    // Deliver with jitter
    delayWithJitter(() => {
      const latency = Date.now() - timestamp;
      stats.latencySum += latency;
      stats.maxLatency = Math.max(stats.maxLatency, latency);
      
      if (type === 'data') {
        // Handle data packet
        stats.packetsReceived++;
        
        // Decode packet
        const result = decoder.decode(packet);
        
        // Send ACK if needed
        if (result.ack) {
          stats.ackTotal++;
          sendAck(result.ack);
        }
      } else {
        // Handle ACK packet
        stats.ackReceived++;
        encoder.handleAck(packet);
      }
    });
  }
}

/**
 * Send data packet
 * 
 * @param {any} state State to encode
 */
function sendDataPacket(state) {
  const packet = encoder.encode(state);
  stats.packetsSent++;
  
  // Add to network queue
  networkQueue.push({
    packet,
    timestamp: Date.now(),
    type: 'data',
    sequence: packet.sequence,
  });
  
  // Process timeouts (would normally be done on a timer)
  const timedOut = encoder.checkAckTimeouts();
  if (timedOut.length > 0) {
    console.log(`[${formatTime()}] ${timedOut.length} packets timed out, retrying...`);
    
    // Re-send timed out packets
    for (const { sequence, state } of timedOut) {
      const resendPacket = encoder.encode(state, true);
      stats.packetsSent++;
      
      // Add to network queue
      networkQueue.push({
        packet: resendPacket,
        timestamp: Date.now(),
        type: 'data',
        sequence: resendPacket.sequence,
      });
    }
  }
  
  // Process network
  processNetworkQueue();
}

/**
 * Send ACK packet
 * 
 * @param {DeltaAck} ack ACK packet
 */
function sendAck(ack) {
  // Add to network queue
  networkQueue.push({
    packet: ack,
    timestamp: Date.now(),
    type: 'ack',
    sequence: ack.sequence,
  });
  
  // Process network
  processNetworkQueue();
}

/**
 * Print statistics
 */
function printStats() {
  const elapsedSec = (Date.now() - stats.testStartTime) / 1000;
  const avgLatency = stats.latencySum / (stats.packetsReceived + stats.ackReceived);
  const packetLossRate = stats.packetsLost / stats.packetsTotal;
  const resyncRate = stats.resyncRequests / elapsedSec;
  const effectiveRate = stats.packetsReceived / elapsedSec;
  
  console.log('\n----- TEST RESULTS -----');
  console.log(`Duration: ${elapsedSec.toFixed(1)} seconds`);
  console.log(`Packets: sent=${stats.packetsSent}, received=${stats.packetsReceived}, lost=${stats.packetsLost} (${(packetLossRate * 100).toFixed(2)}%)`);
  console.log(`ACKs: sent=${stats.ackTotal}, received=${stats.ackReceived}`);
  console.log(`Resyncs: ${stats.resyncRequests} (${resyncRate.toFixed(2)}/sec)`);
  console.log(`Latency: avg=${avgLatency.toFixed(2)}ms, max=${stats.maxLatency}ms`);
  console.log(`Effective packet rate: ${effectiveRate.toFixed(2)} Hz`);
  console.log('------------------------');
  
  // Success criteria from Kaizen ticket KAIZ-001
  if (effectiveRate >= 190) { // 95% of 200 Hz
    console.log('✅ SUCCESS: Achieved target packet rate (>190 Hz)');
  } else {
    console.log('❌ FAILURE: Failed to achieve target packet rate');
  }
  
  if (packetLossRate <= 0.005) { // 0.5%
    console.log('✅ SUCCESS: Packet loss below threshold (<0.5%)');
  } else {
    console.log('❌ FAILURE: Packet loss exceeds threshold');
  }
}

/**
 * Run the test
 */
function runTest() {
  console.log('Starting Delta Protocol resilience test...');
  console.log(`Settings: ${PACKET_RATE_HZ} Hz, ${PACKET_LOSS_RATE * 100}% loss, ${MAX_JITTER_MS}ms jitter`);
  console.log(`Running for ${TEST_DURATION_SEC} seconds...\n`);
  
  // Generate test data (complex enough to exercise delta encoding)
  const testState = {
    counter: 0,
    phases: Array(50).fill(0).map(() => Math.random() * Math.PI * 2),
    timestamp: Date.now(),
    metadata: {
      version: "1.0.0",
      buildHash: "test123",
      settings: {
        mode: "test",
        features: ["delta", "ack", "backoff"]
      }
    }
  };
  
  // Send packets at regular intervals
  const intervalMs = 1000 / PACKET_RATE_HZ;
  const interval = setInterval(() => {
    // Update test state
    testState.counter++;
    testState.timestamp = Date.now();
    
    // Randomly update a few phases
    for (let i = 0; i < 5; i++) {
      const idx = Math.floor(Math.random() * testState.phases.length);
      testState.phases[idx] += Math.random() * 0.1;
      if (testState.phases[idx] >= Math.PI * 2) {
        testState.phases[idx] -= Math.PI * 2;
      }
    }
    
    // Send packet
    sendDataPacket(testState);
    
  }, intervalMs);
  
  // End test after duration
  setTimeout(() => {
    clearInterval(interval);
    
    // Allow time for final packets to be processed
    setTimeout(() => {
      printStats();
    }, MAX_JITTER_MS * 2);
    
  }, TEST_DURATION_SEC * 1000);
}

// Run the test
runTest();
